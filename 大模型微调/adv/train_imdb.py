from peft import LoraConfig, get_peft_model
from transformers import BertForSequenceClassification, BertTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import evaluate
import torch

# 1. 加载数据集（以情感分类数据集 imdb 为例）
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 文本预处理函数
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# 2. 配置 LoraConfig（文本分类专用）
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"],  # BERT 的 attention 层名
    bias="none",
    modules_to_save=["classifier"]  # 全量微调分类头
)

# 3. 加载预训练模型（指定分类任务的类别数，imdb 是 2 分类）
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    torch_dtype=torch.float32  # 节省显存
)

# 4. 注入 LoRA 适配器到模型
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()  # 查看可训练参数比例（通常 < 5%，体现参数效率）

# 5. 配置训练参数
training_args = TrainingArguments(
    output_dir="./lora-bert-imdb",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-4,  # LoRA 学习率通常比全量微调高（1e-4 ~ 3e-4）
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True  # 支持 GPU 时开启混合精度训练
)

# 6. 定义评估指标（准确率）
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return metric.compute(predictions=predictions, references=labels)

# 7. 启动训练
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics
)

trainer.train()

# 8. 保存 LoRA 适配器（仅保存增量参数，体积小）
peft_model.save_pretrained("lora-bert-imdb-adapter")