from transformers import AutoModelForSequenceClassification
loaded_model = AutoModelForSequenceClassification.from_pretrained("./lora_finetuned_model")
print(loaded_model.named_parameters())