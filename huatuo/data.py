from datasets import load_dataset

# 加载训练集、验证集和测试集
data_files = {
    "train": "../train_datasets.jsonl",
    "validation": "../validation_datasets.jsonl",
    "test": "../test_datasets.jsonl"
}

# 使用 load_dataset 加载数据集
dataset = load_dataset("json", data_files=data_files)

# 查看每个分割的数据集
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

# 打印数据集的前5条记录
print("Training set samples:")
print(train_dataset[:5])

print("\nValidation set samples:")
print(validation_dataset[:5])

print("\nTest set samples:")
print(test_dataset[:5])
