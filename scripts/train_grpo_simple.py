"""veRL v0.5.0 简化入口：每20步验证，只保存最后第200步。

从 Base 开始新训练；直接调用官方 run_ppo，不选择验证最优模型。
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

from datetime import datetime
from pathlib import Path

import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.main_ppo import run_ppo

PROJECT = Path("/home/hong/cyan/verl-grpo")
MODEL = "/home/hong/cyan/homework-5/model/Qwen2.5-Math-1.5B"
CONFIG_DIR = PROJECT / "verl/verl/trainer/config"


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
        # 验证仍每20步；保存周期等于总步数，本次正常运行只在第200步保存。
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": 1,
        "trainer.total_epochs": 1,
        "trainer.total_training_steps": 200,
        "trainer.critic_warmup": 0,
        "trainer.val_before_train": True,
        "trainer.test_freq": 20,
        "trainer.save_freq": 200,
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


def main():
    run_dir = PROJECT / "outputs" / datetime.now().strftime(
        "grpo_simple_%Y%m%d_%H%M%S_%f"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    cfg = build_config(run_dir)
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)

    print("输出目录：", run_dir, flush=True)
    print("先验证，再训练200步；每20步验证；只保存最后第200步。", flush=True)

    # Ray 初始化、worker 装配、训练循环均交给官方入口。
    try:
        run_ppo(cfg)
    finally:
        ray.shutdown()

    print("训练完成，最终 checkpoint：",
          run_dir / "checkpoints/global_step_200", flush=True)


if __name__ == "__main__":
    main()
