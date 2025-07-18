import nlpaug.augmenter.word as naw

# 同义词替换增强器
aug = naw.SynonymAug(aug_src='wordnet')

# 对文本进行增强
text = "Data augmentation is important for deep learning."
augmented_text = aug.augment(text)
print("原始文本:", text)
print("增强后文本:", augmented_text)