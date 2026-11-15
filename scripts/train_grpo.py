"""veRL v0.5.0：单卡 Math Base GRPO，按验证正确数只保留最佳 checkpoint。

新训练：python -u train_grpo.py
恢复：  python -u train_grpo.py --resume-run /绝对路径/grpo_formal_...

只有验证/保存策略是项目扩展，GRPO 训练循环仍是 RayPPOTrainer.fit。
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


PROJECT = Path("/home/hong/cyan/verl-grpo")
MODEL = "/home/hong/cyan/homework-5/model/Qwen2.5-Math-1.5B"
CONFIG_DIR = PROJECT / "verl/verl/trainer/config"
VAL_COUNT = 256
VAL_METRIC = "val-core/openai/gsm8k/reward/mean@1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path, value):
    """同一目录写临时文件，写完后原子替换，避免半份 best.json。"""
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def check_checkpoint(folder):
    """核对本项目单卡 checkpoint 的必要文件；不加载大权重。"""
    required = [
        "actor/model_world_size_1_rank_0.pt",
        "actor/optim_world_size_1_rank_0.pt",
        "actor/extra_state_world_size_1_rank_0.pt",
        "actor/fsdp_config.json",
        "actor/huggingface/config.json",
        "actor/huggingface/tokenizer_config.json",
        "data.pt",
    ]
    missing = [name for name in required
               if not (folder / name).is_file() or (folder / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"checkpoint 不完整：{folder}，缺失/空文件：{missing}")


def data_fingerprints(cfg):
    return {
        split: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for split, path in [("train", cfg.data.train_files), ("val", cfg.data.val_files)]
    }


class BestValidationTrainer(RayPPOTrainer):
    """保留官方训练循环，只扩展验证记录和 checkpoint 选择。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint_root = Path(self.config.trainer.default_local_dir)
        self.run_dir = self.checkpoint_root.parent
        self.best_path = self.run_dir / "best.json"
        self.best = read_json(self.best_path) if self.best_path.exists() else None
        self.last_validation = None

    def _validate(self):
        # 官方过程：生成 -> GSM8K 评分 -> 汇总验证指标。
        print(f"[开始验证] step={self.global_steps}，共{VAL_COUNT}题；期间没有参数更新", flush=True)
        metrics = super()._validate()
        if VAL_METRIC not in metrics:
            raise RuntimeError(f"缺少预期验证指标 {VAL_METRIC}；实际：{list(metrics)}")
        accuracy = float(metrics[VAL_METRIC])
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise RuntimeError(f"验证正确率异常：{accuracy}")
        correct = round(accuracy * VAL_COUNT)
        if abs(accuracy * VAL_COUNT - correct) > 1e-4:
            raise RuntimeError("验证指标不能还原成256题的整数正确数，停止选择 checkpoint")

        record = {"step": int(self.global_steps), "correct": correct,
                  "total": VAL_COUNT, "accuracy": accuracy}
        self.last_validation = record
        with (self.run_dir / "validation_history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.global_steps == 0:
            # Base 仅作基线，不参与“训练后最佳”竞争。
            atomic_json(self.run_dir / "base_validation.json", record)
        print(f"[验证] step={self.global_steps}: {correct}/{VAL_COUNT} = {accuracy:.2%}",
              flush=True)
        return metrics

    def _save_checkpoint(self):
        result = self.last_validation
        # 同一训练步必须先完成验证，才允许参与最佳选择。
        if result is None or result["step"] != self.global_steps or self.global_steps == 0:
            return
        if self.best is not None and result["correct"] <= self.best["correct"]:
            print(f"[保存] 未严格超过最佳 {self.best['correct']}/{VAL_COUNT}，不保存", flush=True)
            return

        # 不使用官方 keep=1：必须先成功保存新最佳，再删除旧最佳。
        super()._save_checkpoint()
        folder = self.checkpoint_root / f"global_step_{self.global_steps}"
        check_checkpoint(folder)
        new_best = {**result, "checkpoint_path": str(folder)}
        atomic_json(self.best_path, new_best)
        self.best = new_best
        print(f"[新最佳] {result['correct']}/{VAL_COUNT}，已保存：{folder}", flush=True)

        # 只清理本次运行 checkpoints/ 下的其他 step 目录。
        # 新保存失败、必要文件检查失败时，程序不会走到这里。
        for old in self.checkpoint_root.glob("global_step_*"):
            if old == folder or old.is_symlink() or not old.is_dir():
                continue
            if not old.name.removeprefix("global_step_").isdigit():
                continue
            if old.resolve().parent != self.checkpoint_root.resolve():
                raise RuntimeError(f"拒绝清理越界路径：{old}")
            shutil.rmtree(old)
            print(f"[清理] 已删除旧 checkpoint：{old}", flush=True)


def build_config(run_dir):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="ppo_trainer")

    # 键名与官方 YAML 一一对应。force_add 允许添加 model_dtype 等扩展字段。
    settings = {
        # 数据：沿用准备好的消息与划分，训练入口不读取独立 test.parquet。
        "data.train_files": str(PROJECT / "data/gsm8k/train.parquet"),
        "data.val_files": str(PROJECT / "data/gsm8k/val.parquet"),
        "data.train_batch_size": 8,
        "data.val_batch_size": 256,
        "data.max_prompt_length": 1024,
        "data.max_response_length": 1024,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.dataloader_num_workers": 0,
        "data.seed": 42,
        "data.shuffle": True,
        "data.validation_shuffle": False,
        # 模型。
        "actor_rollout_ref.model.path": MODEL,
        "actor_rollout_ref.model.lora_rank": 0,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        # 训练生成：8题 × 4回答，随机采样。
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "sync",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.dtype": "bfloat16",
        "actor_rollout_ref.rollout.n": 4,
        "actor_rollout_ref.rollout.do_sample": True,
        "actor_rollout_ref.rollout.temperature": 1.0,
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.max_model_len": 2048,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 2048,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.3,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.enable_chunked_prefill": False,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.seed": 42,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs": 4,
        # 验证生成：greedy，每题只有1个回答。
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        # 官方规则奖励、GRPO，无参考模型 KL。
        "reward_model.enable": False,
        "custom_reward_function.path": None,
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.use_kl_in_reward": False,
        # BF16 全参数更新，无 CPU offload。
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "bf16",
        "actor_rollout_ref.actor.fsdp_config.offload_policy": False,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.optim.lr": 1e-5,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.optim.warmup_style": "constant",
        "actor_rollout_ref.actor.ppo_mini_batch_size": 8,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.actor.checkpoint.load_contents": ["model", "optimizer", "extra"],
        # 周期验证后，由 BestValidationTrainer 决定本步是否真的保存。
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": 1,
        "trainer.total_epochs": 1,
        "trainer.total_training_steps": 200,
        "trainer.critic_warmup": 0,
        "trainer.val_before_train": True,
        "trainer.test_freq": 20,
        "trainer.save_freq": 20,
        "trainer.resume_mode": "disable",
        "trainer.del_local_ckpt_after_load": False,
        "trainer.max_actor_ckpt_to_keep": None,
        "trainer.max_critic_ckpt_to_keep": None,
        "trainer.remove_previous_ckpt_in_save": False,
        "trainer.logger": ["console"],
        "trainer.project_name": "verl_grpo_learning",
        "trainer.experiment_name": run_dir.name,
        "trainer.default_local_dir": str(run_dir / "checkpoints"),
        "trainer.default_hdfs_dir": None,
        "trainer.rollout_data_dir": str(run_dir / "rollouts"),
        "trainer.validation_data_dir": str(run_dir / "validation"),
        "ray_init.num_cpus": 4,
    }
    for key, value in settings.items():
        OmegaConf.update(cfg, key, value, force_add=True)
    return cfg


@ray.remote(num_cpus=1)
class TrainingTaskRunner:
    """采用官方 TaskRunner 的装配方式，只注册本次实际需要的 ActorRollout。"""

    def run(self, cfg):
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.single_controller.ray import RayWorkerGroup
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.workers.fsdp_workers import ActorRolloutRefWorker

        OmegaConf.resolve(cfg)
        if (cfg.algorithm.adv_estimator != "grpo" or cfg.algorithm.use_kl_in_reward
                or cfg.actor_rollout_ref.actor.use_kl_loss or cfg.reward_model.enable
                or cfg.actor_rollout_ref.actor.strategy != "fsdp2"
                or cfg.actor_rollout_ref.rollout.mode != "sync"):
            raise ValueError("本入口只支持已确认的 FSDP2、同步 rollout、规则奖励、无 KL 的 GRPO")

        tokenizer = hf_tokenizer(cfg.actor_rollout_ref.model.path, trust_remote_code=False)
        processor = hf_processor(cfg.actor_rollout_ref.model.path,
                                 trust_remote_code=False, use_fast=True)
        train_data = create_rl_dataset(cfg.data.train_files, cfg.data, tokenizer,
                                       processor, is_train=True)
        val_data = create_rl_dataset(cfg.data.val_files, cfg.data, tokenizer,
                                     processor, is_train=False)
        if len(train_data) != 7217 or len(val_data) != VAL_COUNT:
            raise ValueError(f"数据题数变化/发生过滤：train={len(train_data)}, val={len(val_data)}")

        # 官方验证按解码后的输入分组；防止重复输入改变 mean@1 的含义。
        decoded = [tokenizer.decode(val_data[i]["input_ids"], skip_special_tokens=True)
                   for i in range(len(val_data))]
        if len(set(decoded)) != VAL_COUNT:
            raise ValueError("验证集含重复的实际输入，不能直接按 mean@1 选择最佳")

        pool = ResourcePoolManager(
            resource_pool_spec={"single_gpu": [1]},
            mapping={Role.ActorRollout: "single_gpu"},
        )
        trainer = BestValidationTrainer(
            config=cfg,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping={Role.ActorRollout: ray.remote(ActorRolloutRefWorker)},
            resource_pool_manager=pool,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=load_reward_manager(cfg, tokenizer, num_examine=0),
            val_reward_fn=load_reward_manager(cfg, tokenizer, num_examine=0),
            train_dataset=train_data,
            val_dataset=val_data,
            collate_fn=collate_fn,
            train_sampler=create_rl_sampler(cfg.data, train_data),
        )
        trainer.init_workers()
        trainer.fit()
        if trainer.best is None:
            raise RuntimeError("训练结束但没有生成最佳 checkpoint")
        return trainer.best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-run", type=Path, default=None,
                        help="已有正式运行目录，从该目录 best.json 指定的最佳步恢复")
    args = parser.parse_args()

    if args.resume_run is None:
        run_dir = PROJECT / "outputs" / datetime.now().strftime("grpo_formal_%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
        cfg = build_config(run_dir)
        OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
        atomic_json(run_dir / "data_fingerprints.json", data_fingerprints(cfg))
        shutil.copyfile(Path(__file__).resolve(), run_dir / "train_grpo_source.py")
    else:
        run_dir = args.resume_run.expanduser().resolve()
        if (run_dir / "training_complete.json").exists():
            raise ValueError("这次运行已完成200步，无需恢复；最佳模型信息见 best.json")
        cfg = OmegaConf.load(run_dir / "config.yaml")
        best = read_json(run_dir / "best.json")
        folder = Path(best["checkpoint_path"]).resolve()
        expected = run_dir / "checkpoints" / f"global_step_{int(best['step'])}"
        if folder != expected or Path(cfg.trainer.default_local_dir).resolve() != run_dir / "checkpoints":
            raise ValueError("运行目录与 best.json/config.yaml 不一致，不支持直接移动运行目录后恢复")
        if data_fingerprints(cfg) != read_json(run_dir / "data_fingerprints.json"):
            raise ValueError("训练或验证数据已变化，拒绝混用旧 checkpoint 恢复")
        check_checkpoint(folder)
        if best["step"] >= cfg.trainer.total_training_steps:
            raise ValueError("最佳 checkpoint 已达到总训练步数，无需继续")
        cfg.trainer.resume_mode = "resume_path"
        cfg.trainer.resume_from_path = str(folder)
        cfg.trainer.val_before_train = False
        OmegaConf.save(cfg, run_dir / "resume_config.yaml", resolve=True)

    print("运行目录：", run_dir, flush=True)
    print("200步；8题×4回答；BF16；无 offload；micro=1；固定学习率1e-5", flush=True)
    print("验证：训练前及每20步，256题 greedy；只保留训练后的最佳 checkpoint", flush=True)
    print("注意：启动后先做完整验证，训练进度条会在初始验证完成后出现。", flush=True)
    ray.init(runtime_env=get_ppo_ray_runtime_env(), num_cpus=cfg.ray_init.num_cpus)
    try:
        runner = TrainingTaskRunner.remote()
        best = ray.get(runner.run.remote(cfg))
        atomic_json(run_dir / "training_complete.json", {"total_steps": 200, "best": best})
        print("\n========== 正式训练完成 ==========", flush=True)
        print(json.dumps(best, ensure_ascii=False, indent=2), flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
