# 任务类型	推荐格式	示例
# 问答系统	指令格式	输入问题，输出答案
# 文本生成	纯文本格式	给定开头，续写全文
# 对话系统	对话格式	多轮对话历史
# 文本分类	指令格式	"判断情感" + 文本 + 情感标签
# 机器翻译	指令格式(带input字段)	输入原文，输出译文
# 指令微调
instruction = [
    {
        "instruction": "解释量子计算的基本原理",
        "input": "",  # 可选的上下文或补充信息
        "output": "量子计算是利用量子力学原理..."
    },
    {
        "instruction": "将以下句子翻译成英文",
        "input": "今天天气真好",
        "output": "The weather is nice today."
    }
]
# 对话格式
conversations = [
    {
        "conversations": [
            {"role": "user", "content": "量子计算是什么？"},
            {"role": "assistant", "content": "量子计算是利用量子力学原理..."}
        ]
    },
    {
        "conversations": [
            {"role": "user", "content": "请用英文说'今天天气真好'"},
            {"role": "assistant", "content": "The weather is nice today."}
        ]
    }
]