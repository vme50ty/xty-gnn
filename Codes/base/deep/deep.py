import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 用户类
class User:
    def __init__(self, user_id, belong_proxy=1, is_attacker=False):
        self.id = user_id
        self.requestCount = 0
        self.avg_latency = 0
        self.requestPerMinute = 0
        self.connsCount = random.randint(0, 5)
        self.credit = 0
        self.belong = belong_proxy
        self.is_attacker = is_attacker

    def generate_behavior(self):
        """普通用户的行为生成"""
        self.requestCount = random.randint(0, 50)
        self.avg_latency = random.uniform(0.1, 1)
        self.requestPerMinute = random.randint(1, 10)
        self.credit = random.randint(1, 8)

class DirectAttacker(User):
    def generate_behavior(self):
        """直接攻击者：高请求数量，高信用"""
        self.requestCount = random.randint(1000, 2000)
        self.avg_latency = random.uniform(0.5, 2)
        self.requestPerMinute = random.randint(50, 200)
        self.credit = random.randint(5, 10)


class SlowAttacker(User):
    def generate_behavior(self):
        """慢速攻击者：随机提高两个指标"""
        super().generate_behavior()  # 先生成普通用户的行为
        
        # 可被提升的指标
        possible_features = ["requestCount", "avg_latency", "requestPerMinute", "connsCount"]
        chosen_features = random.sample(possible_features, 2)  # 随机选择两个指标提高
        
        if "requestCount" in chosen_features:
            self.requestCount = random.randint(30,50)  # 提高请求数

        if "avg_latency" in chosen_features:
            self.avg_latency = random.uniform(0.5, 2)  # 提高时延

        if "requestPerMinute" in chosen_features:
            self.requestPerMinute = random.randint(5, 15)  # 提高请求速率
        
        if "connsCount" in chosen_features:
            self.connsCount = random.randint(5, 8)  # 增加并发连接

class StealthAttacker(User):
    def generate_behavior(self):
        """高隐蔽攻击者：与普通用户行为相同，但所在代理异常"""
        super().generate_behavior()

# 生成数据集
def generate_dataset(num_users=3000):
    data = []
    labels = []
    
    # 生成普通用户
    for i in range(int(num_users *0.97)):
        user = User(i)
        user.generate_behavior()
        data.append([user.requestCount, user.avg_latency, user.requestPerMinute, user.connsCount, user.credit])
        labels.append(0)  # 0 表示正常用户
    
    # 生成攻击者（每种类型占 1/6）
    for i in range(int(num_users *0.01)):
        attacker = DirectAttacker(i + num_users)
        attacker.generate_behavior()
        data.append([attacker.requestCount, attacker.avg_latency, attacker.requestPerMinute, attacker.connsCount, attacker.credit])
        labels.append(1)  # 1 表示直接攻击者

        attacker = SlowAttacker(i + num_users * 2)
        attacker.generate_behavior()
        data.append([attacker.requestCount, attacker.avg_latency, attacker.requestPerMinute, attacker.connsCount, attacker.credit])
        labels.append(2)  # 2 表示慢速攻击者

        attacker = StealthAttacker(i + num_users * 3)
        attacker.generate_behavior()
        data.append([attacker.requestCount, attacker.avg_latency, attacker.requestPerMinute, attacker.connsCount, attacker.credit])
        labels.append(3)  # 3 表示高隐蔽攻击者

    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)

# 加载数据
X, y = generate_dataset(3000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch Tensor
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# **构建 ResNet-18 模型**
class ResNetClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=4):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        
        # 修改输入通道数（原始 ResNet 适用于 3 通道图片）
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 修改全连接层
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

        self.fc_input = nn.Linear(input_dim, in_features)

    def forward(self, x):
        x = F.relu(self.fc_input(x))  # 映射输入特征到 ResNet 兼容维度
        x = x.unsqueeze(1).unsqueeze(1)  # 适配 ResNet 维度
        x = self.resnet(x)
        return x

# 初始化模型
model = ResNetClassifier(input_dim=X_train.shape[1]).to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
def train_model(model, train_loader, optimizer, criterion, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# 训练和评估
train_model(model, train_loader, optimizer, criterion, epochs=100)
evaluate_model(model, test_loader)
