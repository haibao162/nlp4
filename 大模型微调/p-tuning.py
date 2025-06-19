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
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
from trl import SFTTrainer

# 设置设备（GPU/CPU）
# device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载模型和分词器
model_path = "/Users/mac/Documents/DeepSeek-R1-Distill-Qwen-1.5B"
model_name = "deepseek-ai/deepseek-r1-distill-qwen1.5-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # 设置padding token

# 2. 量化配置（可选，降低显存占用）
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,          # 4-bit量化
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",  # 4-bit NormalFloat
#     bnb_4bit_compute_dtype=torch.bfloat16
# )

# 3. 加载模型（可选择是否量化）
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # quantization_config=bnb_config,  # 如果显存不足，启用量化
    device_map="auto",              # 自动分配GPU/CPU
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# 4. 准备模型（如果用了4-bit量化）
# model = prepare_model_for_kbit_training(model)

# 5. 配置LoRA
lora_config = LoraConfig(
    r=8,                  # LoRA秩（Rank）
    lora_alpha=32,        # Alpha参数（缩放因子）
    target_modules=["q_proj"],  # 作用的目标层
    lora_dropout=0.05,    # Dropout率
    bias="none",          # 是否训练偏置
    task_type="CAUSAL_LM"  # 任务类型（因果语言模型）
)

# 6. 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 打印可训练参数数量

# 7. 加载数据集
dataset = load_dataset("json", data_files="data.json", split="train")

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
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()  # 训练时计算loss
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = dataset.map(lambda x: {"text": f"{x['instruction']}\n{x['output']}"})


# 9. 训练参数
training_args = TrainingArguments(
    output_dir="./lora_finetuned_model",  # 保存路径
    per_device_train_batch_size=4,       # 批次大小
    gradient_accumulation_steps=4,       # 梯度累积
    num_train_epochs=3,                  # 训练轮次
    learning_rate=2e-4,                  # 学习率
    fp16=True,                           # 混合精度训练
    logging_steps=10,                    # 每10步打印日志
)

# 10. 创建Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    # data_collator=lambda data: {
    #     "input_ids": torch.stack([torch.tensor(item["input_ids"]) for item in data]),
    #     "attention_mask": torch.stack([torch.tensor(item["attention_mask"]) for item in data]),
    #     "labels": torch.stack([torch.tensor(item["labels"]) for item in data]),
    # },
)

# 11. 开始训练！
trainer.train()

# 12. 保存模型（LoRA权重）
model.save_pretrained("./lora_finetuned_model")
tokenizer.save_pretrained("./lora_finetuned_model")