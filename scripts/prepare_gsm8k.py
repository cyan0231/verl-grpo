"""准备使用 few-shot 提示词的 veRL GSM8K 数据。"""

import json
import random
import re
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


PROJECT_DIR = Path("/home/hong/cyan/verl-grpo")
SOURCE_DIR = Path("/home/hong/cyan/homework-5/data/gsm8k")
OUTPUT_DIR = PROJECT_DIR / "data/gsm8k"

MODEL_PATH = "/home/hong/cyan/homework-5/model/Qwen2.5-Math-1.5B"

VAL_SIZE = 256
SPLIT_SEED = 42

# 与刚才 check_rollout_reward.py 中的提示词保持一致。
FORMAT_SYSTEM = (
    "Solve the math problem step by step. "
    "End your response with a final line in exactly this format:\n"
    "#### <answer>\n"
    "Replace <answer> with the numerical answer. "
    "Do not use \\boxed{}. "
    "Do not write anything after the final answer line."
)

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

QUESTION_SUFFIX = (
    "\nSolve step by step and end with #### followed "
    "by the numerical answer, as shown in the examples."
)


def read_jsonl(path):
    """读取原始 GSM8K，提前检查字段和标准答案。"""
    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            item = json.loads(line)

            for field in ("question", "answer"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"{path} 第 {line_number} 行的 {field} "
                        "必须是非空字符串"
                    )

            # 写文件前就发现标准答案异常。
            extract_ground_truth(item["answer"])
            records.append(item)

    if not records:
        raise ValueError(f"数据文件为空：{path}")

    return records


def extract_ground_truth(answer):
    """从原始答案中提取 #### 后的数值，保留字符串形式。"""
    matches = re.findall(r"#### (\-?[0-9\.\,]+)", answer)

    if not matches:
        raise ValueError(
            f"原始答案缺少有效的 #### 数值：{answer!r}"
        )

    return matches[-1].replace(",", "")


def build_prompt(question):
    """构造消息列表；不在这里渲染聊天模板或进行 tokenization。"""
    return [
        {"role": "system", "content": FORMAT_SYSTEM},
        *[dict(message) for message in FORMAT_EXAMPLES],
        {
            "role": "user",
            "content": question + QUESTION_SUFFIX,
        },
    ]


def build_row(item, source_index, split):
    """转换成 veRL 默认数据加载器需要的字段。"""
    question = item["question"]
    answer = item["answer"]

    return {
        # 默认评分分发入口据此选择官方 GSM8K 评分函数。
        "data_source": "openai/gsm8k",

        # 只有格式示例和实际问题，不包含实际问题的标准答案。
        "prompt": build_prompt(question),

        "ability": "math",

        # 标准答案供奖励函数使用。
        "reward_model": {
            "style": "rule",
            "ground_truth": extract_ground_truth(answer),
        },

        # index 保留原始文件中的样本序号，不随划分重新编号。
        "extra_info": {
            "split": split,
            "index": source_index,
            "question": question,
            "answer": answer,
        },
    }


def save_parquet(rows, filename):
    path = OUTPUT_DIR / filename
    Dataset.from_list(rows).to_parquet(str(path))

    # 读回检查，确认文件可以正常加载。
    restored = Dataset.from_parquet(str(path))

    if len(restored) != len(rows):
        raise RuntimeError(f"写入前后样本数量不一致：{path}")

    if restored[0]["prompt"] != rows[0]["prompt"]:
        raise RuntimeError(f"写入前后第一条 prompt 不一致：{path}")

    print(f"{path.stem}：{len(restored)} 题 → {path}")
    return restored


def main():
    raw_train = read_jsonl(SOURCE_DIR / "train.jsonl")
    raw_test = read_jsonl(SOURCE_DIR / "test.jsonl")

    if len(raw_train) <= VAL_SIZE:
        raise ValueError(
            f"训练集只有 {len(raw_train)} 题，"
            f"无法划分 {VAL_SIZE} 题验证集并保留训练数据"
        )

    # 保留之前的划分算法：
    # random.Random(42).sample，而不是改用 datasets.shuffle。
    validation_indices = set(
        random.Random(SPLIT_SEED).sample(
            range(len(raw_train)),
            VAL_SIZE,
        )
    )

    train_rows = []
    val_rows = []

    # 保持原始样本顺序。
    for source_index, item in enumerate(raw_train):
        if source_index in validation_indices:
            val_rows.append(
                build_row(item, source_index, "validation")
            )
        else:
            train_rows.append(
                build_row(item, source_index, "train")
            )

    test_rows = [
        build_row(item, source_index, "test")
        for source_index, item in enumerate(raw_test)
    ]

    # 检查训练、验证划分完整且没有索引重叠。
    train_indices = {
        row["extra_info"]["index"] for row in train_rows
    }
    val_indices = {
        row["extra_info"]["index"] for row in val_rows
    }

    assert train_indices.isdisjoint(val_indices)
    assert len(train_rows) + len(val_rows) == len(raw_train)
    assert len(val_rows) == VAL_SIZE

    # 先验证 tokenizer 和模板可用，再写出文件。
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )
    tokenizer.apply_chat_template(
        train_rows[0]["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_dataset = save_parquet(train_rows, "train.parquet")
    save_parquet(val_rows, "val.parquet")
    save_parquet(test_rows, "test.parquet")

    manifest = {
        "seed": SPLIT_SEED,
        "source_train_count": len(raw_train),
        "source_test_count": len(raw_test),
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "test_count": len(test_rows),
        "validation_source_indices": sorted(validation_indices),
        "prompt_version": "gsm8k_hash4_fewshot_v1",
        "model_path": MODEL_PATH,
    }

    manifest_path = OUTPUT_DIR / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 展示从 Parquet 读回的数据经模型模板渲染后的实际输入。
    first = train_dataset[0]
    rendered = tokenizer.apply_chat_template(
        first["prompt"],
        tokenize=False,
        add_generation_prompt=True,
    )
    token_ids = tokenizer.encode(
        rendered,
        add_special_tokens=False,
    )

    print("\n========== 第一条训练样本实际输入 ==========")
    print(rendered)
    print("输入 token 数：", len(token_ids))
    print("标准答案（不在实际问题的输入中）：",
          first["reward_model"]["ground_truth"])
    print("\n划分记录已保存：", manifest_path)
    print("数据准备完成。")


if __name__ == "__main__":
    main()