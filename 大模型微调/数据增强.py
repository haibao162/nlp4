from textattack.augmentation import WordNetAugmenter
augmenter = WordNetAugmenter()
augmented_text = augmenter.augment("今天天气真好")
print(augmented_text)