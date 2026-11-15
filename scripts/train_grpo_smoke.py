"""veRL 两步 GRPO 测试：BF16 参数、无 CPU offload、micro-batch=1。"""

import os
from datetime import datetime
from pathlib import Path

# 在导入涉及 CUDA 的模块之前设置。
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"

# 保持正常异步执行，不开启同步调试。
os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl.trainer.main_ppo import run_ppo


PROJECT_DIR = Path("/home/hong/cyan/verl-grpo")
MODEL_PATH = "/home/hong/cyan/homework-5/model/Qwen2.5-Math-1.5B"
CONFIG_DIR = PROJECT_DIR / "verl/verl/trainer/config"


def build_config(run_dir):
    # 读取并合并 veRL 官方默认配置。
    with initialize_config_dir(
        config_dir=str(CONFIG_DIR),
        version_base=None,
    ):
        cfg = compose(config_name="ppo_trainer")

    # ------------------------------------------------------------
    # 1. 数据
    # ------------------------------------------------------------
    cfg.data.train_files = str(
        PROJECT_DIR / "data/gsm8k/train.parquet"
    )
    cfg.data.val_files = str(
        PROJECT_DIR / "data/gsm8k/val.parquet"
    )

    # 每步 8 道题，few-shot 示例已包含在 Parquet 的 prompt 中。
    cfg.data.train_batch_size = 8
    cfg.data.max_prompt_length = 1024
    cfg.data.max_response_length = 1024
    cfg.data.filter_overlong_prompts = True
    cfg.data.truncation = "error"
    cfg.data.dataloader_num_workers = 0

    OmegaConf.update(
        cfg,
        "data.seed",
        42,
        force_add=True,
    )

    # ------------------------------------------------------------
    # 2. 模型
    # ------------------------------------------------------------
    model = cfg.actor_rollout_ref.model
    model.path = MODEL_PATH

    # 全参数训练。
    model.lora_rank = 0
    model.use_remove_padding = True
    model.enable_gradient_checkpointing = True

    # ------------------------------------------------------------
    # 3. 生成
    # ------------------------------------------------------------
    rollout = cfg.actor_rollout_ref.rollout
    rollout.name = "vllm"
    rollout.mode = "sync"
    rollout.tensor_model_parallel_size = 1
    rollout.dtype = "bfloat16"

    # 每题 4 个回答，每步合计 32 个回答。
    rollout.n = 4
    rollout.temperature = 1.0
    rollout.top_p = 1.0
    rollout.top_k = -1

    rollout.max_model_len = 2048
    rollout.max_num_batched_tokens = 2048
    rollout.gpu_memory_utilization = 0.3
    rollout.enforce_eager = True
    rollout.free_cache_engine = True
    rollout.enable_chunked_prefill = False

    # 计算旧策略 log probability 时，每次处理 1 条回答。
    rollout.log_prob_micro_batch_size_per_gpu = 1

    OmegaConf.update(
        cfg,
        "actor_rollout_ref.rollout.seed",
        42,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.max_num_seqs",
        4,
        force_add=True,
    )

    # ------------------------------------------------------------
    # 4. 奖励和优势
    # ------------------------------------------------------------
    # 使用数据来源 openai/gsm8k 对应的官方规则评分。
    cfg.reward_model.enable = False
    cfg.custom_reward_function.path = None

    cfg.algorithm.adv_estimator = "grpo"
    cfg.algorithm.norm_adv_by_std_in_grpo = True
    cfg.algorithm.use_kl_in_reward = False

    # ------------------------------------------------------------
    # 5. 参数更新
    # ------------------------------------------------------------
    actor = cfg.actor_rollout_ref.actor
    actor.strategy = "fsdp2"

    # 本次修改：训练参数存储为 BF16。
    # 当前普通 AdamW 路径的动量状态也会随参数采用 BF16。
    OmegaConf.update(
        cfg,
        "actor_rollout_ref.actor.fsdp_config.model_dtype",
        "bf16",
        force_add=True,
    )

    # 本次修改：关闭 CPU offload。
    actor.fsdp_config.offload_policy = False
    actor.fsdp_config.param_offload = False
    actor.fsdp_config.optimizer_offload = False

    actor.optim.lr = 1e-6

    # worker 内会乘以 rollout.n：
    # 8 × 4 = 32 条回答，组成一次优化器更新。
    actor.ppo_mini_batch_size = 8

    # 每次前向、反向处理 1 条回答，累积 32 次。
    actor.ppo_micro_batch_size_per_gpu = 1
    actor.ppo_epochs = 1

    # 使用官方默认损失聚合方式。
    actor.loss_agg_mode = "token-mean"

    actor.use_kl_loss = False
    actor.entropy_coeff = 0.0
    actor.use_torch_compile = False

    # ------------------------------------------------------------
    # 6. 两步测试与输出
    # ------------------------------------------------------------
    trainer = cfg.trainer
    trainer.nnodes = 1
    trainer.n_gpus_per_node = 1

    trainer.total_epochs = 1
    trainer.total_training_steps = 2
    trainer.critic_warmup = 0

    # 本次只测试训练流程和资源占用。
    trainer.val_before_train = False
    trainer.test_freq = -1
    trainer.save_freq = -1

    # 每次从原始 Base 开始，不恢复之前的小实验。
    trainer.resume_mode = "disable"

    trainer.logger = ["console"]
    trainer.project_name = "verl_grpo_learning"
    trainer.experiment_name = run_dir.name
    trainer.default_local_dir = str(run_dir / "checkpoints")
    trainer.rollout_data_dir = str(run_dir / "rollouts")

    cfg.ray_init.num_cpus = 4

    return cfg


def main():
    run_name = datetime.now().strftime(
        "grpo_smoke_%Y%m%d_%H%M%S_%f"
    )
    run_dir = PROJECT_DIR / "outputs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    cfg = build_config(run_dir)

    config_path = run_dir / "config.yaml"
    OmegaConf.save(
        config=cfg,
        f=str(config_path),
        resolve=True,
    )

    print("模型：", cfg.actor_rollout_ref.model.path)
    print(
        "训练参数精度：",
        cfg.actor_rollout_ref.actor.fsdp_config.model_dtype,
    )
    print(
        "CPU offload：",
        cfg.actor_rollout_ref.actor.fsdp_config.offload_policy,
    )
    print(
        "训练 micro-batch：",
        cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
    )
    print("每步题数：", cfg.data.train_batch_size)
    print("每题回答数：", cfg.actor_rollout_ref.rollout.n)
    print("训练步数：", cfg.trainer.total_training_steps)
    print("配置文件：", config_path)
    print("输出目录：", run_dir, flush=True)

    # 进入 veRL 的实际训练流程。
    run_ppo(cfg)


if __name__ == "__main__":
    main()