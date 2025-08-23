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
    {"role": "assistant", "content": "帮我分析输入的句子在游戏中是否为广告拉人，例如加v12333代表加微信，是广告拉人，输出概率，并严格按照JSON格式返回结果"},
    {"role": "user", "content": """
    请回答以下问题，并严格按照以下JSON格式返回，不要添加任何额外内容：{{"answer": "[回答内容]", "prob": "[拉人概率]"}}
    问题：加v12333
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





