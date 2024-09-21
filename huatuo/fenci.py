import jieba
from collections import Counter
from datasets import load_dataset

# 加载训练集、验证集和测试集
data_files = {
    "train": "../train_datasets.jsonl",
    "validation": "../validation_datasets.jsonl",
    "test": "../test_datasets.jsonl"
}

# 使用 load_dataset 加载数据集
dataset = load_dataset("json", data_files=data_files)

# 查看训练集的数据
train_dataset = dataset['train']

# 初始化计数器
word_counter = Counter()

# 对每一条记录进行分词并统计词频
for sample in train_dataset:
    for question in sample['questions']:
        # 对问题进行分词
        words = jieba.lcut(question)
        # 统计所有词语的出现次数
        word_counter.update(words)

# 打印最常见的前10个词语及其出现次数
print("最常见的词语：")
for word, count in word_counter.most_common(10):
    print(f"{word}: {count}")

# 如果需要输出更多词频信息，可以调整 .most_common() 的参数
