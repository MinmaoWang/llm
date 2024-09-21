from datasets import load_dataset

dataset = load_dataset("tau/commonsense_qa")

"""
    train: Dataset({
        features: ['id', 'question', 'question_concept', 'choices', 'answerKey'],
        num_rows: 9741
    })
"""
print(dataset)
# 查看每个分割的数据集
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

print(train_dataset['question'][:10])

print(set(train_dataset['question_concept']))
print(len(set(train_dataset['question_concept'])))
print(len(train_dataset['choices']))