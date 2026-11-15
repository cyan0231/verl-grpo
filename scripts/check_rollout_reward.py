"""使用 few-shot 提示词检查 Math Base 的生成和官方 GSM8K 奖励。"""

import json
import os
from pathlib import Path

# 必须在导入 vLLM 之前设置。
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"

from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.gsm8k import extract_solution


MODEL_PATH = "/home/hong/cyan/homework-5/model/Qwen2.5-Math-1.5B"
DATA_PATH = "/home/hong/cyan/verl-grpo/data/gsm8k/train.parquet"
OUTPUT_PATH = Path(
    "/home/hong/cyan/verl-grpo/outputs/rollout_check_32x4_fewshot.json"
)

NUM_QUESTIONS = 32
RESPONSES_PER_QUESTION = 4
MAX_NEW_TOKENS = 1024
MAX_MODEL_LEN = 2048
SEED = 42

FORMAT_SYSTEM = (
    "Solve the math problem step by step. "
    "End your response with a final line in exactly this format:\n"
    "#### <answer>\n"
    "Replace <answer> with the numerical answer. "
    "Do not use \\boxed{}. "
    "Do not write anything after the final answer line."
)

# 这两个示例用于演示格式，不包含实际待测题目的答案。
FORMAT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Tom has 3 apples and buys 2 more. "
            "How many apples does he have?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Tom has 3 + 2 = 5 apples.\n"
            "#### 5"
        ),
    },
    {
        "role": "user",
        "content": (
            "A notebook costs 4 dollars. "
            "How much do 3 notebooks cost?"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "The total cost is 4 × 3 = 12 dollars.\n"
            "#### 12"
        ),
    },
]


def main():
    # 与上一版使用相同的抽样方式和种子，保持题目一致。
    dataset = Dataset.from_parquet(DATA_PATH)
    if len(dataset) < NUM_QUESTIONS:
        raise ValueError(
            f"数据集只有 {len(dataset)} 题，"
            f"不足以抽取 {NUM_QUESTIONS} 题"
        )

    dataset = dataset.shuffle(seed=SEED).select(range(NUM_QUESTIONS))
    samples = [dataset[i] for i in range(len(dataset))]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    prompts = []
    rendered_prompts = []

    for question_index, sample in enumerate(samples, start=1):
        if sample["data_source"] != "openai/gsm8k":
            raise ValueError(
                f"非预期的数据来源：{sample['data_source']}"
            )

        # 使用新的 system 指令和格式示例。
        # 实际问题只取 question，不把标准答案放进输入。
        messages = [
            {"role": "system", "content": FORMAT_SYSTEM},
            *FORMAT_EXAMPLES,
            {
                "role": "user",
                "content": (
                    sample["extra_info"]["question"]
                    + "\nSolve step by step and end with #### followed "
                    "by the numerical answer, as shown in the examples."
                ),
            },
        ]

        # 仍然使用模型自带的 Qwen 聊天模板。
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(token_ids) + MAX_NEW_TOKENS > MAX_MODEL_LEN:
            raise ValueError(
                f"第 {question_index} 题超过上下文预算："
                f"输入 {len(token_ids)} tokens + "
                f"最多生成 {MAX_NEW_TOKENS} tokens "
                f"> {MAX_MODEL_LEN}"
            )

        if question_index == 1:
            print("========== 第一题实际输入 ==========")
            print(text)
            print("输入 token 数：", len(token_ids), flush=True)

        prompts.append({"prompt_token_ids": token_ids})
        rendered_prompts.append(text)

    # 这里只进行生成和评分，不创建训练优化器。
    llm = LLM(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=4,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
        seed=SEED,
    )

    params = SamplingParams(
        n=RESPONSES_PER_QUESTION,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED,
    )

    results = llm.generate(
        prompts,
        sampling_params=params,
    )

    if len(results) != len(samples):
        raise RuntimeError("生成结果数量与输入题目数量不一致")

    records = []
    group_rewards = []

    for question_index, (sample, result) in enumerate(
        zip(samples, results),
        start=1,
    ):
        if len(result.outputs) != RESPONSES_PER_QUESTION:
            raise RuntimeError(
                f"第 {question_index} 题没有生成预期数量的回答"
            )

        ground_truth = sample["reward_model"]["ground_truth"]
        rewards = []

        print(f"\n========== 题目 {question_index} ==========")
        print(sample["extra_info"]["question"])
        print("标准答案：", ground_truth)

        for response_index, output in enumerate(result.outputs, start=1):
            # 使用 veRL 官方入口及 GSM8K 默认 strict 评分。
            reward = float(
                default_compute_score(
                    data_source=sample["data_source"],
                    solution_str=output.text,
                    ground_truth=ground_truth,
                    extra_info=sample["extra_info"],
                )
            )
            extracted_answer = extract_solution(
                output.text,
                method="strict",
            )

            record = {
                "question_index": question_index,
                "source_index": sample["extra_info"]["index"],
                "response_index": response_index,
                "question": sample["extra_info"]["question"],
                "rendered_prompt": rendered_prompts[question_index - 1],
                "ground_truth": ground_truth,
                "response": output.text,
                "generated_tokens": len(output.token_ids),
                "finish_reason": output.finish_reason,
                "extracted_answer": extracted_answer,
                "reward": reward,
            }
            records.append(record)
            rewards.append(reward)

            print(f"\n--- 回答 {response_index} ---")
            print(output.text)
            print("生成 token 数：", record["generated_tokens"])
            print("结束原因：", record["finish_reason"])
            print("提取答案：", extracted_answer)
            print("奖励：", reward)

        group_rewards.append(rewards)

    # 保持原来的记录列表结构，之前的读取命令仍可使用。
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    parsed_count = sum(
        r["extracted_answer"] is not None for r in records
    )
    correct_count = sum(r["reward"] == 1.0 for r in records)
    truncated_count = sum(
        r["finish_reason"] == "length" for r in records
    )
    positive_question_count = sum(
        any(reward > 0 for reward in rewards)
        for rewards in group_rewards
    )
    mixed_count = sum(
        len(set(rewards)) > 1 for rewards in group_rewards
    )
    all_zero_count = sum(
        all(reward == 0 for reward in rewards)
        for rewards in group_rewards
    )
    all_one_count = sum(
        all(reward == 1 for reward in rewards)
        for rewards in group_rewards
    )

    print("\n========== 每题奖励 ==========")
    for question_index, rewards in enumerate(group_rewards, start=1):
        print(
            f"题目 {question_index:02d}：{rewards}"
            f" | 组内奖励有差异：{len(set(rewards)) > 1}"
        )

    print("\n========== 检查结果 ==========")
    print("题目数：", len(samples))
    print("回答总数：", len(records))
    print("官方规则可提取答案的数量：", parsed_count)
    print("获得正确答案奖励的数量：", correct_count)
    print(f"正奖励回答比例：{correct_count / len(records):.2%}")
    print("至少一个回答获得正奖励的题数：", positive_question_count)
    print("组内奖励有差异的题数：", mixed_count)
    print("全部零分的题数：", all_zero_count)
    print("全部一分的题数：", all_one_count)
    print("达到生成长度上限的回答数：", truncated_count)
    print("完整结果已保存：", OUTPUT_PATH)


if __name__ == "__main__":
    main()