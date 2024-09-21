import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# 加载数据集
dataset = load_dataset("tau/commonsense_qa")

train_dataset = dataset['train']
validation_dataset = dataset['validation']

# 定义常量
BATCH_SIZE = 32
MAX_SEQ_LEN = 128
LEARNING_RATE = 1e-4
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载 BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 自定义数据集类
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        choices = self.data[idx]['choices']['text']
        answer_key = ord(self.data[idx]['answerKey']) - ord('A')  # 将 'A', 'B', 'C', 'D', 'E' 转换为 0-4

        # 将问题与每个选项拼接后编码
        inputs = [self.tokenizer.encode_plus(question + " " + choice, max_length=self.max_len, padding='max_length',
                                             truncation=True, return_tensors="pt") for choice in choices]

        # 将输入张量化
        input_ids = torch.cat([inp['input_ids'] for inp in inputs])
        attention_mask = torch.cat([inp['attention_mask'] for inp in inputs])

        return {
            'input_ids': input_ids,  # (num_choices, max_len)
            'attention_mask': attention_mask,
            'label': torch.tensor(answer_key)
        }


# 定义多层感知器模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 使用 BERT 提取嵌入并训练 MLP
class CommonsenseQAClassifier(nn.Module):
    def __init__(self, hidden_size=128):
        super(CommonsenseQAClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.mlp = MLP(self.bert.config.hidden_size, hidden_size, num_classes=5)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            # 获取 BERT 的最后一层 [CLS] token 输出
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        logits = self.mlp(outputs)
        return logits


# 数据加载器
def create_data_loader(data, tokenizer, max_len, batch_size):
    ds = QADataset(data, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


train_loader = create_data_loader(train_dataset, tokenizer, MAX_SEQ_LEN, BATCH_SIZE)
valid_loader = create_data_loader(validation_dataset, tokenizer, MAX_SEQ_LEN, BATCH_SIZE)

# 初始化模型、损失函数、优化器
model = CommonsenseQAClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# 训练函数
def train_model(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0

    for data in data_loader:
        input_ids = data['input_ids'].to(device).view(-1, MAX_SEQ_LEN)  # 处理为 (batch_size * num_choices, max_len)
        attention_mask = data['attention_mask'].to(device).view(-1, MAX_SEQ_LEN)
        labels = data['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        outputs = outputs.view(-1, 5)  # (batch_size, num_choices)

        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)

        loss.backward()
        optimizer.step()

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader.dataset)


# 验证函数
def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for data in data_loader:
            input_ids = data['input_ids'].to(device).view(-1, MAX_SEQ_LEN)
            attention_mask = data['attention_mask'].to(device).view(-1, MAX_SEQ_LEN)
            labels = data['label'].to(device)

            outputs = model(input_ids, attention_mask)
            outputs = outputs.view(-1, 5)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return correct_predictions.double() / len(data_loader.dataset), total_loss / len(data_loader.dataset)


# 主训练循环
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    train_acc, train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE)
    print(f"Train loss {train_loss}, accuracy {train_acc}")

    val_acc, val_loss = eval_model(model, valid_loader, criterion, DEVICE)
    print(f"Validation loss {val_loss}, accuracy {val_acc}")
