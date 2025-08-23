from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, AutoModelForSequenceClassification
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import time
import torch
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

model_path = "/Users/nn/Documents/yjx/bert-base-chinese"

# 加载数据集
dataset = load_dataset("json", data_files={"train": "train_data.json", "test": "test_data.json"})
print(dataset)
# DatasetDict({
#     train: Dataset({
#         features: ['text', 'label'],
#         num_rows: 3
#     })
# })

# 加载模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, 
num_labels=2, 
trust_remote_code=True,
torch_dtype=torch.float32,
device_map="auto"
)

# 输入转换为模型输入格式
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
# 应用预处理函数到整个数据集
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# 准备训练参数
training_args = TrainingArguments(
    # output_dir="./lora-bert-imdb",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,  # LoRA 学习率通常比全量微调高（1e-4 ~ 3e-4）
    num_train_epochs=20,
    eval_strategy="epoch",
    # save_strategy="epoch",
    # logging_dir="./logs",
    # logging_steps=10,
)

# 5. 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 6. 初始化Trainer并训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# 执行训练 - 这一步会更新分类器的权重
trainer.train()

# 7. 保存训练好的模型（包含已训练的分类器权重）
model.save_pretrained("./trained_bert_classifier")
tokenizer.save_pretrained("./trained_bert_classifier")
