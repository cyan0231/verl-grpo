"""同一测试集评测 Base 与 GRPO 最佳模型；独立子进程依次使用单张 GPU。"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT = Path("/home/hong/cyan/verl-grpo")
BASE = Path("/home/hong/cyan/homework-5/model/Qwen2.5-Math-1.5B")
TRAINED = PROJECT / (
    "outputs/grpo_formal_20261113_205104_280464/hf_model_step_160"
)
DATA = PROJECT / "data/gsm8k/test.parquet"
MODELS = {"base": BASE, "grpo_step_160": TRAINED}
NUM_QUESTIONS = 1319
SAMPLING = {
    "n": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
    "min_p": 0.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
    "repetition_penalty": 1.0, "max_tokens": 1024, "min_tokens": 0,
    "ignore_eos": False, "seed": 42,
}
ENGINE = {
    "dtype": "bfloat16", "tensor_parallel_size": 1,
    "max_model_len": 2048, "max_num_seqs": 64,
    "max_num_batched_tokens": 2048, "gpu_memory_utilization": 0.9,
    "enforce_eager": True, "enable_chunked_prefill": False, "seed": 42,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_one(label, out_dir):
    # 主进程不导入 CUDA 库；子进程在这里初始化独立的推理环境。
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

    from importlib.metadata import version
    from datasets import Dataset
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score import default_compute_score
    from verl.utils.reward_score.gsm8k import extract_solution

    manifest = read_json(out_dir / "comparison_config.json")
    if fingerprint(DATA) != manifest["test_sha256"]:
        raise RuntimeError("测试数据文件发生变化，停止对比")
    if fingerprint(BASE / "generation_config.json") != manifest["generation_config_sha256"]:
        raise RuntimeError("基准生成配置发生变化，停止对比")
    tokenizer = AutoTokenizer.from_pretrained(str(BASE), local_files_only=True)
    dataset = Dataset.from_parquet(str(DATA))
    if len(dataset) != NUM_QUESTIONS:
        raise ValueError(f"测试集应为1319题，实际为{len(dataset)}题")

    samples, prompts = [], []
    input_digest = hashlib.sha256()
    for i, sample in enumerate(dataset):
        if sample["data_source"] != "openai/gsm8k":
            raise ValueError(f"第{i}题不是 openai/gsm8k")
        text = tokenizer.apply_chat_template(
            sample["prompt"], tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) > 1024:
            raise ValueError(f"第{i}题输入超过1024 tokens，停止而非截断")
        input_digest.update((json.dumps(ids) + "\n").encode("utf-8"))
        samples.append(sample)
        prompts.append({"prompt_token_ids": ids})

    print(f"\n========== 开始评测 {label}：1319题 ==========", flush=True)
    # 两者共用 Base tokenizer 和 generation_config，固定输入编码与停止配置。
    # 权重路径是两次评测的主要变量。
    llm = LLM(
        model=str(MODELS[label]), tokenizer=str(BASE),
        generation_config=str(BASE), **ENGINE,
    )
    results = llm.generate(prompts, sampling_params=SamplingParams(**SAMPLING), use_tqdm=True)
    if len(results) != NUM_QUESTIONS:
        raise RuntimeError("生成结果数量不完整")

    correct = parsed = truncated = total_tokens = 0
    with (out_dir / f"{label}_responses.jsonl").open("w", encoding="utf-8") as f:
        for i, (sample, prompt, result) in enumerate(zip(samples, prompts, results)):
            if len(result.outputs) != 1 or list(result.prompt_token_ids) != prompt["prompt_token_ids"]:
                raise RuntimeError(f"第{i}题生成数量/输入不符合预期")
            output = result.outputs[0]
            gold = sample["reward_model"]["ground_truth"]
            score = float(default_compute_score(
                data_source=sample["data_source"], solution_str=output.text,
                ground_truth=gold, extra_info=sample["extra_info"],
            ))
            if score not in (0.0, 1.0):
                raise RuntimeError(f"官方二值奖励出现异常值：{score}")
            extracted = extract_solution(output.text, method="strict")
            correct += int(score == 1.0)
            parsed += int(extracted is not None)
            truncated += int(output.finish_reason == "length")
            total_tokens += len(output.token_ids)
            record = {
                "row_index": i, "source_index": sample["extra_info"]["index"],
                "question": sample["extra_info"]["question"], "ground_truth": gold,
                "response": output.text, "extracted_answer": extracted, "reward": score,
                "prompt_tokens": len(prompt["prompt_token_ids"]),
                "generated_tokens": len(output.token_ids), "finish_reason": output.finish_reason,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    metrics = {
        "model": label, "model_path": str(MODELS[label]), "num_questions": NUM_QUESTIONS,
        "correct": correct, "accuracy": correct / NUM_QUESTIONS,
        "extractable_answers": parsed, "extractable_rate": parsed / NUM_QUESTIONS,
        "length_limited_answers": truncated, "length_limited_rate": truncated / NUM_QUESTIONS,
        "mean_generated_tokens": total_tokens / NUM_QUESTIONS,
        "prompt_token_ids_sha256": input_digest.hexdigest(),
        "versions": {name: version(name) for name in ("torch", "vllm", "transformers", "verl")},
    }
    write_json(out_dir / f"{label}_metrics.json", metrics)
    print(f"\n[{label}] 正确：{correct}/{NUM_QUESTIONS} = {metrics['accuracy']:.2%}", flush=True)
    print(f"[{label}] 可提取答案：{parsed}；达到长度上限：{truncated}", flush=True)


def summarize(out_dir):
    before = read_json(out_dir / "base_metrics.json")
    after = read_json(out_dir / "grpo_step_160_metrics.json")
    for key in ("num_questions", "prompt_token_ids_sha256", "versions"):
        if before[key] != after[key]:
            raise RuntimeError(f"两次评测的 {key} 不一致，拒绝汇总")
    if before["num_questions"] != NUM_QUESTIONS:
        raise RuntimeError("题数不是1319")

    def read_records(label):
        with (out_dir / f"{label}_responses.jsonl").open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    base_records, trained_records = read_records("base"), read_records("grpo_step_160")
    if len(base_records) != NUM_QUESTIONS or len(trained_records) != NUM_QUESTIONS:
        raise RuntimeError("逐题文件不完整")
    paired = {"both_correct": 0, "both_wrong": 0, "wrong_to_correct": 0, "correct_to_wrong": 0}
    with (out_dir / "paired_results.jsonl").open("w", encoding="utf-8") as f:
        for a, b in zip(base_records, trained_records):
            for key in ("row_index", "source_index", "question", "ground_truth"):
                if a[key] != b[key]:
                    raise RuntimeError(f"逐题对齐失败：{key}")
            ac, bc = a["reward"] == 1.0, b["reward"] == 1.0
            category = ("both_correct" if ac and bc else "correct_to_wrong" if ac
                        else "wrong_to_correct" if bc else "both_wrong")
            paired[category] += 1
            f.write(json.dumps({"category": category, "base": a, "grpo": b}, ensure_ascii=False) + "\n")

    if paired["both_correct"] + paired["correct_to_wrong"] != before["correct"]:
        raise RuntimeError("Base 汇总与逐题成绩不一致")
    if paired["both_correct"] + paired["wrong_to_correct"] != after["correct"]:
        raise RuntimeError("GRPO 汇总与逐题成绩不一致")
    delta = (after["accuracy"] - before["accuracy"]) * 100
    summary = {"base": before, "grpo_step_160": after,
               "accuracy_delta_percentage_points": delta, "paired": paired}
    write_json(out_dir / "comparison_summary.json", summary)
    print("\n========== 独立测试集对比完成 ==========", flush=True)
    for label, m in [("Base", before), ("GRPO step160", after)]:
        print(f"{label}: {m['correct']}/{NUM_QUESTIONS} = {m['accuracy']:.2%}")
        print(f"  可提取答案：{m['extractable_rate']:.2%}；长度上限：{m['length_limited_rate']:.2%}；"
              f"平均生成 tokens：{m['mean_generated_tokens']:.1f}")
    print(f"正确率变化：{delta:+.2f} 个百分点")
    print(f"由错变对：{paired['wrong_to_correct']}；由对变错：{paired['correct_to_wrong']}")
    print("汇总文件：", out_dir / "comparison_summary.json", flush=True)


def stop_evaluation():
    """只停止同一脚本的评测进程树，不触碰训练或其他 GPU 作业。"""
    from contextlib import suppress
    import psutil

    script = str(Path(__file__).resolve())
    roots = []
    for p in psutil.process_iter(["cmdline"]):
        cmd = p.info["cmdline"] or []
        if p.pid != os.getpid() and script in cmd and "--stop" not in cmd:
            roots.append(p)
    targets = {}
    try:
        # 控制进程先暂停，避免清理时又启动下一个模型。
        for p in roots:
            with suppress(psutil.NoSuchProcess):
                p.suspend()
        for p in roots:
            with suppress(psutil.NoSuchProcess):
                targets[p.pid] = p
                for child in p.children(recursive=True):
                    targets[child.pid] = child
        for p in targets.values():
            with suppress(psutil.NoSuchProcess):
                p.terminate()
    finally:
        for p in roots:
            with suppress(psutil.NoSuchProcess):
                p.resume()
    _, alive = psutil.wait_procs(list(targets.values()), timeout=10)
    for p in alive:
        with suppress(psutil.NoSuchProcess):
            p.kill()
    _, alive = psutil.wait_procs(alive, timeout=10)
    if alive:
        raise RuntimeError(f"仍有进程未退出：{[p.pid for p in alive]}")
    print("已停止本脚本的旧评测进程：", sorted(targets), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", action="store_true", help="停止这份脚本正在运行的评测及子进程")
    parser.add_argument("--worker", choices=list(MODELS), help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stop:
        stop_evaluation()
        return
    if args.worker:
        if args.output_dir is None:
            parser.error("worker 模式需要 output-dir")
        evaluate_one(args.worker, args.output_dir)
        return

    # 先检查两份模型均存在，避免测完 Base 才发现导出目录有误。
    for label, path in MODELS.items():
        if not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
            raise FileNotFoundError(f"{label} 模型配置或权重不存在：{path}")
    base_cfg, trained_cfg = read_json(BASE / "config.json"), read_json(TRAINED / "config.json")
    # BOS 元数据可能被 veRL 按 tokenizer 改写；本脚本传入非空 prompt_token_ids，
    # 不依赖模型配置自动插入 BOS，因此不要求两份 config 的 BOS 字段相等。
    for key in ("model_type", "vocab_size", "eos_token_id"):
        if base_cfg.get(key) != trained_cfg.get(key):
            raise ValueError(f"模型配置 {key} 不一致，请先核对导出模型")
    out_dir = PROJECT / "outputs" / datetime.now().strftime("compare_base_grpo_%Y%m%d_%H%M%S_%f")
    out_dir.mkdir(parents=True, exist_ok=False)
    write_json(out_dir / "comparison_config.json", {
        "models": {label: str(path) for label, path in MODELS.items()},
        "test_path": str(DATA), "test_sha256": fingerprint(DATA),
        "tokenizer_path": str(BASE), "generation_config_path": str(BASE),
        "generation_config_sha256": fingerprint(BASE / "generation_config.json"),
        "sampling": SAMPLING, "engine": ENGINE, "num_questions": NUM_QUESTIONS,
    })
    print("对比输出目录：", out_dir, flush=True)
    for label in MODELS:
        # 不使用 shell；等前一个评测进程退出，再启动下一个。
        subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()),
                        "--worker", label, "--output-dir", str(out_dir)], check=True)
    summarize(out_dir)


if __name__ == "__main__":
    main()
