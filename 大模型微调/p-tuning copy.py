# from datasets import Dataset

# dataset = Dataset.from_dict({
#     "instruction": ["解释量子计算", "翻译成英文"],
#     "input": ["", "今天天气真好"],
#     "output": ["量子计算是利用...", "The weather is nice today."]
# })

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os

# 设置设备（GPU/CPU）
# device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载模型和分词器
model_path = "/Users/mac/Documents/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "deepseek-ai/deepseek-r1-distill-qwen1.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # 设置padding token


# 7. 加载数据集
dataset = load_dataset("json", data_files="data.json", split="train")
result = dataset.map(lambda x: {"text": f"{x['instruction']}\n{x['output']}"})
for i in result:
    print(i, 'result')


# 8. 数据预处理
def preprocess_function(examples):
    inputs = [
        f"Instruction: {inst}\nInput: {inp}\nOutput: {out}"
        for inst, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"]
        )
    ]
    # print(inputs, 'inputs')
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()  # 训练时计算loss
    # return model_inputs

preprocess_function(dataset)

