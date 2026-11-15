# veRL GRPO：单卡 GSM8K 学习实验

使用固定历史版本 **veRL v0.5.0**，在单张 RTX 3090 24GB 上，从 **Qwen2.5-Math-1.5B Base** 开始进行全参数 GRPO 训练。

本项目用于学习数据准备、生成、规则奖励、组内优势、参数更新和模型评测。不包含 Search-R1、在线搜索或工具调用，也不是官方基准复现。

## 实验结果

同一份独立 GSM8K 测试集，相同提示词、tokenizer 和生成配置：greedy，每题一个回答，最多生成 1024 token，使用 veRL 官方 GSM8K 严格评分。

| 指标 | Math Base | GRPO 第 160 步 |
| --- | ---: | ---: |
| 正确数 | 7/1319 | 1016/1319 |
| 正确率 | 0.53% | **77.03%** |
| 可提取答案比例 | 8.42% | 99.39% |
| 达到生成长度上限 | 97.19% | 0.45% |
| 平均生成 token 数 | 1005.6 | 95.4 |

正确率提高 **76.50 个百分点**；1009 题由错变对，0 题由对变错。训练共完成 200 步，按验证集选择第 160 步模型，其验证成绩为 222/256 = 86.72%。验证集用于选模型，独立测试集用于报告最终成绩。

- [评测配置](outputs/compare_base_grpo_20261114_012259_919915/comparison_config.json)
- [结果汇总](outputs/compare_base_grpo_20261114_012259_919915/comparison_summary.json)

这是一次实验记录。Base 在当前提示词和 greedy 解码下存在严重的重复、格式和停止问题，因此 **0.53% 不代表它一般意义上的数学解题能力**，也不能把全部提升归因于新增推理能力。可提取率不是严格格式合规率，长度截断率也不是重复率。

仓库只保留代码、准备好的数据和最终评测汇总；没有模型权重、训练 checkpoint、完整训练日志或逐题回答。结果来自已完成的服务器实验，本地整理没有重新训练或评测。训练步数和最佳验证成绩来自当时的运行记录，完整验证历史未随仓库保存。

## 文件结构

```text
verl-grpo/
├── README.md
├── THIRD_PARTY_NOTICES.md
├── scripts/
│   ├── prepare_gsm8k.py        # 数据划分、few-shot 提示词、Parquet
│   ├── check_rollout_reward.py # 训练前抽样生成与奖励诊断
│   ├── train_grpo_smoke.py     # 两步训练，检查流程和显存
│   ├── train_grpo.py           # 正式训练，按验证成绩保留最佳模型
│   ├── train_grpo_simple.py    # 简化入口，直接调用官方 run_ppo
│   └── compare_models.py       # Base 与训练模型的独立测试对比
├── data/gsm8k/
│   ├── train.parquet
│   ├── val.parquet
│   ├── test.parquet
│   ├── split_manifest.json    # 固定验证集索引与提示词版本
│   └── LICENSE                # GSM8K 数据许可证
├── outputs/
│   └── compare_base_grpo_20261114_012259_919915/
│       ├── comparison_config.json
│       └── comparison_summary.json
└── verl/                      # 上游框架 Git 子模块，固定 v0.5.0
```

## 训练配置

| 项目 | 配置 |
| --- | --- |
| 起点 | Qwen2.5-Math-1.5B Base，未经过本项目 SFT |
| 数据划分 | 训练 7217、验证 256、独立测试 1319 |
| 划分方法 | 从原始训练集 7473 题中用 `random.Random(42).sample` 抽取验证集 |
| 提示词 | 显式 system、两个 few-shot 示例、当前问题；最终答案为 `#### 数值` |
| 更新方式 | 全参数、FSDP2、BF16 参数、梯度检查点、无 CPU offload |
| 训练规模 | 200 步，每步 8 题 × 4 回答 = 32 条回答 |
| mini-batch / micro-batch | 配置值 8 / 1；veRL 将 mini-batch 按回答数扩展为 32 条 |
| 学习率 | 固定 `1e-5`，无 warmup、无衰减 |
| 优势 / 损失 | GRPO 组内标准差归一化；`token-mean` |
| KL / 熵奖励 | 不使用参考模型 KL，熵系数为 0 |
| 长度上限 | prompt 1024、response 1024、总上下文 2048 token |
| 训练采样 | `temperature=1.0`、`top_p=1.0`、每题 4 次 |
| 验证 | 训练前及每 20 步，greedy，每题一次 |
| checkpoint | 验证正确数严格提高才替换最佳模型；平分保留旧模型 |

具体设置见 [正式训练入口](scripts/train_grpo.py)。这里显式采用 BF16 参数以适应单卡资源，不能等同于 FP32 参数加 BF16 混合精度的配置。

`train_grpo_simple.py` 仍然每 20 步验证，但只在最后第 200 步保存，不执行最佳模型选择。它便于学习官方入口，**不是上述第 160 步成绩的训练入口**。

## 使用方式

### 获取项目

```bash
git clone --recurse-submodules https://github.com/cyan0231/verl-grpo.git
cd verl-grpo
```

如果克隆时没有下载子模块，执行 `git submodule update --init --recursive`。

`verl/` 固定在 v0.5.0 的提交 `8fdc4d3f202f41461f4de9f42a637228e342668b`，不自动跟随上游最新版。

### 环境和路径

实验使用 Linux、单张 RTX 3090 24GB 和独立 Conda 环境 `verl-grpo`。已运行过的主要版本如下；这不是完整依赖锁文件，仓库没有提供经过验证的一键安装流程。

| 组件 | 版本 |
| --- | --- |
| Python | 3.10 |
| veRL | 0.5.0 |
| PyTorch | 2.6.0+cu124 |
| vLLM | 0.8.5 |
| FlashAttention | 2.7.4.post1 |
| Transformers | 4.51.3 |
| Ray | 2.43.0 |

安装参考 [v0.5.0 官方说明](https://github.com/verl-project/verl/blob/v0.5.0/docs/start/install.rst)，并核对 PyTorch、vLLM、FlashAttention 的二进制兼容性。服务器 Toolkit 为 CUDA 12.8，本次 PyTorch CUDA runtime 为 12.4；`nvidia-smi` 显示的 CUDA Version 不是 Toolkit 版本。

依赖安装完成后，在项目根目录将这份固定源码注册到当前 Python 环境：

```bash
python -m pip install -e ./verl --no-deps
```

`--no-deps` 不会安装缺失依赖；此步骤的用途是让 `import verl` 指向仓库中的源码。

模型需另行准备：[Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)。仓库不分发权重。

脚本保留服务器实验的绝对路径，换机器前需要修改顶部常量：

- `PROJECT` / `PROJECT_DIR`：项目目录，原值 `/home/hong/cyan/verl-grpo`。
- `MODEL` / `MODEL_PATH` / `BASE`：本地 Math Base 模型目录。
- `SOURCE_DIR`：仅重新准备数据时使用，目录内包含原始 `train.jsonl`、`test.jsonl`。
- `TRAINED`：评测时使用的、已导出的 Hugging Face 模型目录。
- 诊断脚本中的 `DATA_PATH`、`OUTPUT_PATH`：对应本机的数据与结果路径。

历史结果 JSON 中的绝对路径用于记录当时的实验，不表示模型已包含在仓库中，不需要为了换机器修改历史记录。

### 数据与训练

仓库已提供 Parquet，可直接用于训练。只有修改提示词或重新准备数据时，才需要从 [GSM8K 原始数据](https://github.com/openai/grade-school-math/tree/master/grade_school_math/data) 准备 JSONL，然后执行：

```bash
python scripts/prepare_gsm8k.py
```

该命令会覆盖三个 Parquet 和划分记录。保持固定划分，避免将独立测试集用于选模型。

先运行两步检查，再正式训练：

```bash
python -u scripts/train_grpo_smoke.py
python -u scripts/train_grpo.py
```

若仅学习配置与官方训练入口，可阅读或单独运行 `scripts/train_grpo_simple.py`。

### 导出与评测

正式训练在输出目录保存 `best.json`，其中记录最佳 checkpoint 路径。先把 actor checkpoint 导出成推理后端可以加载的模型，将以下示例路径替换为实际路径：

```bash
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "/path/to/checkpoint/actor" \
  --target_dir "/path/to/exported_model"
```

修改 `compare_models.py` 中的 `TRAINED` 后执行：

```bash
python -u scripts/compare_models.py
```

两个模型通过独立子进程依次使用同一张 GPU，使用相同测试数据和提示词，生成新的配置、逐题结果和汇总。停止该脚本管理的评测进程：

```bash
python scripts/compare_models.py --stop
```

独立评测配置使用并发上限 64、显存预算 0.9；训练生成使用并发上限 4、显存预算 0.3。评测不同时保留训练优化器，不能把评测显存配置直接套到训练中。

## 阅读顺序

1. [prepare_gsm8k.py](scripts/prepare_gsm8k.py)：原始题目如何变成 prompt、标准答案和 Parquet。
2. [train_grpo_simple.py](scripts/train_grpo_simple.py)：配置如何交给 `run_ppo(cfg)`。
3. `verl/verl/trainer/main_ppo.py`：组装 worker、数据与奖励。
4. `verl/verl/trainer/ppo/ray_trainer.py`：生成 → 评分 → 优势 → 更新 → 验证。
5. `verl/verl/trainer/ppo/core_algos.py`：GRPO 优势与策略损失。
6. [train_grpo.py](scripts/train_grpo.py)：最佳 checkpoint 的选择与保存。
7. [compare_models.py](scripts/compare_models.py)：同一评测规则下比较两个模型。

## 维护与结果解释

- 新实验使用新输出目录，记录配置、数据划分与模型来源；不要覆盖已报告的结果。
- 改提示词、停止规则或评分方式时，应重新评测 Base 与训练模型，不能混用不同协议的分数。
- `.gitignore` 默认排除新运行输出、日志、权重和 checkpoint；确认需要公开的新结果后，再显式加入版本管理。
- 旧手写 GRPO 记录为 1061/1319 = 80.44%，比本次多对 45 题。但旧实验从 SFT 模型开始，每步 32×8 回答，评测使用随机采样和 `<answer>` 评分；本次起点、采样量和评测协议均不同，不能据此判断框架优劣。

## 来源

- [veRL](https://github.com/verl-project/verl)：以固定 Git 子模块引用，遵循其 Apache-2.0 许可证。
- [GSM8K](https://github.com/openai/grade-school-math)：对原始问答添加提示词、划分与 veRL 数据字段；数据许可证见 [data/gsm8k/LICENSE](data/gsm8k/LICENSE)。
- [第三方材料说明](THIRD_PARTY_NOTICES.md)。模型权重需遵循模型发布方的许可。
