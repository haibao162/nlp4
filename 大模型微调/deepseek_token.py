from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/Users/mac/Documents/DeepSeek-R1-Distill-Qwen-1.5B"
# DeepSeek-R1-Distill-Qwen-1.5B
# model_name = "Qwen/Qwen-1.5B-Chat"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,  # 自动选择精度
    trust_remote_code=True,
    device_map="auto", # 自动处理设备分配
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

prompt = "请介绍一下人工智能的发展历程。"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
# {'input_ids': tensor([[151646,  14880, 109432, 104455, 103949, 103168,   1773]]), 
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}
with torch.no_grad():
    outputs = model(**inputs, max_length=200)

    outputs2 = model.generate(
        **inputs,
        max_length=300,  # 总长度不超过100个token
        temperature=0.7
    )
    

# print(outputs[0].shape, 'outputs')
# torch.Size([1, 7, 151936])
# result = torch.argmax(outputs[0], dim=-1)
# tensor([[   692,  56568, 101888,   9909, 105044,   3837,  14880]])
# result = tokenizer.decode(result[0])
# 你关于（现状，请


print(tokenizer.decode(outputs2[0]), 'outputs2')





