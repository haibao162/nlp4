from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/Users/nn/Documents/yjx/DeepSeek-R1-Distill-Qwen-1.5B"
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

conversation = [
    {"role": "assistant", "content": "我是一个游戏客服，帮我分析输入的句子在游戏中是否为广告拉人"},
    {"role": "user", "content": """
    需要分析的句子：
    加v12333
    """},
]

prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

# 编码输入
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=1000,  # 总长度不超过100个token
        temperature=0.1
    )

print(tokenizer.decode(outputs[0]), 'outputs')





