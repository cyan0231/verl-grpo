# 第三方材料说明

## veRL

`verl/` 通过 Git 子模块引用 [verl-project/verl](https://github.com/verl-project/verl)，固定版本 v0.5.0，提交 `8fdc4d3f202f41461f4de9f42a637228e342668b`。上游代码遵循 Apache-2.0，许可证随子模块保留。

## GSM8K

`data/gsm8k/*.parquet` 包含来自 [OpenAI GSM8K](https://github.com/openai/grade-school-math) 的题目与答案。本项目进行了训练/验证划分，添加 few-shot 提示词及 veRL 所需字段。原始数据采用 MIT 许可证，Copyright (c) 2021 OpenAI；完整许可文本见 [data/gsm8k/LICENSE](data/gsm8k/LICENSE)。

论文：Karl Cobbe et al., [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168), 2021。

## Qwen

脚本使用 [Qwen2.5-Math-1.5B](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B)，仓库不分发模型权重。模型的下载、使用和衍生权重分发须遵循模型发布方的许可。
