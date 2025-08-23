from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def predict_sentiment(text, model, tokenizer, device):
    """
    :param text: 输入文本
    :param model: 加载好的模型
    :param tokenizer: 对应的tokenizer
    :param device: 运行设备（cpu或gpu）
    :return: 预测标签和置信度
    """
    # 文本预处理
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"  # 返回PyTorch张量
    ).to(device)  # 移动到指定设备
    
    # 模型推理（关闭梯度计算，提高速度）
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 解析结果
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)  # 转换为概率
    predicted_label = torch.argmax(probabilities, dim=1).item()  # 获取预测标签
    confidence = probabilities[0][predicted_label].item()  # 获取预测置信度
    
    return predicted_label, confidence

if __name__ == "__main__":
    # 1. 配置参数
    model_dir = "./trained_bert_classifier"  # 训练好的模型目录
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 2. 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)  # 移动模型到指定设备
    
    # 3. 待预测的文本
    test_texts = [
        "群szr12323232",
        "这个不好用",
        "加微信1223141132",
        "v1223141132"

    ]
    
    # 4. 批量预测并输出结果
    for text in test_texts:
        label, confidence = predict_sentiment(text, model, tokenizer, device)
        # 假设0=负面，1=正面，可根据实际标签体系调整
        sentiment = "正面" if label == 1 else "负面"
        print(f"文本: {text}")
        print(f"预测结果: {sentiment} (置信度: {confidence:.4f})")
        print("---")
