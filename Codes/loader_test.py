'''
Author: lee12345 15116908166@163.com
Date: 2024-12-16 10:12:28
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-16 17:00:41
FilePath: /Gnn/DHGNN-LSTM/Codes/loader_test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
import torch.nn as nn
import torch.optim as optim
from src import GraphDataLoader,CombinedModel,SequenceEncoder,Config

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

config=Config()
       
ip_encoder = SequenceEncoder(device='cuda:0' if torch.cuda.is_available() else 'cpu',model_path='/home/lzy/Gnn/DHGNN-LSTM/Codes/src/model/')

encoders1 = {
    'ip': ip_encoder  # 将 IP 列的编码器传入
}
encoders2 = {
    'name': ip_encoder  # 将 IP 列的编码器传入
}

path=config.dataPath

# 初始化 GraphDataLoader
print("Initializing data loader...")
dataLoader1=GraphDataLoader(path,encoders1,encoders2)
dataLoader1.process_data()
train_loader=dataLoader1.get_train_loader()
valid_loader=dataLoader1.get_valid_loader()

# 定义超参数
epochs = 20

time_deltas = [0, 10, 10]  # 时间差分参数

# 初始化模型
print("Initializing model...")
device=config.device
model = CombinedModel(config.input_dim, config.hidden_dim)
model.to(device)


# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

optimizer.zero_grad()
model.train()
total_loss = 0.0
    
# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        
        # 获取输入数据
        graphs, ips, labels = batch 
        graphs = [graph.to(device) for graph in graphs]
        ips = [
            {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in ip_dict.items()}
            for ip_dict in ips
        ]       
        
        # 前向传播
        outputs,global_ips = model(time_deltas,graphs, ips)
        
        # 获取标签
        labels_list = []
        for ip in global_ips:
            label = labels[ip]
            labels_list.append(label)
        labels_tensor = torch.tensor(labels_list).to(device)
        
        loss = criterion(outputs, labels_tensor)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

# 验证函数
def validate_model(model, valid_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in valid_loader:
            graphs, ips, labels = batch 
            graphs = [graph.to(device) for graph in graphs]
            ips = [
                {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in ip_dict.items()}
                for ip_dict in ips
            ]    

            # 前向传播
            outputs,global_ips = model(time_deltas,graphs, ips)
            
            # 获取标签
            labels_list = []
            for ip in global_ips:
                label = labels[ip]
                labels_list.append(label)
            labels_tensor = torch.tensor(labels_list).to(device)
            
            loss = criterion(outputs, labels_tensor)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels_tensor).sum().item()
            total += labels_tensor.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / total
    return avg_loss, accuracy

print("Starting training...")
for epoch in range(epochs):
    train_loss = train_model(model, train_loader, criterion, optimizer, device)
    valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, device)

    print(f"Epoch {epoch + 1}/{epochs}, "
          f"Train Loss: {train_loss:.4f}, "
          f"Validation Loss: {valid_loss:.4f}, "
          f"Validation Accuracy: {valid_accuracy:.4f}")

print("Training completed!")