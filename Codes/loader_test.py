'''
Author: lee12345 15116908166@163.com
Date: 2024-12-16 10:12:28
LastEditors: vme50ty 15116908166@163.com
LastEditTime: 2025-03-13 09:34:44
FilePath: /Gnn/DHGNN-LSTM/Codes/loader_test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
import torch.nn as nn
import torch.optim as optim
from src import GraphDataLoader,CombinedModel,SequenceEncoder,Config
import time,os

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

config=Config()

path=config.dataPath

# 初始化 GraphDataLoader
print("Initializing data loader...")
dataLoader1=GraphDataLoader(path,None,None)
dataLoader1.process_data()
train_loader=dataLoader1.get_train_loader()
valid_loader=dataLoader1.get_valid_loader()

# 定义超参数
epochs = config.epochs

# 初始化模型
print("Initializing model...")
device=config.device
model = CombinedModel(config.input_dim, config.hidden_dim)
model.to(device)

# 定义损失函数和优化器
class_weights = torch.tensor(config.class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        graphs, ips, labels, time_deltas= batch 
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
            graphs, ips, labels,time_deltas = batch 
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
            # print(f'predicted={predicted}')
            # print(f'true_label={labels_tensor}')
            correct += (predicted == labels_tensor).sum().item()
            total += labels_tensor.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(valid_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def save_config(config, save_path):
    """保存配置到文件"""
    with open(save_path, 'w') as f:
        for key, value in config.__dict__.items():
            f.write(f"{key}: {value}\n")


time_now = time.strftime("%Y%m%d-%H%M%S")
result_dir=f'../result/{time_now}'
os.makedirs(result_dir, exist_ok=True)

# 将config存储在resultTest文件
save_config(config, os.path.join(result_dir, 'config.txt'))

log_file_path = os.path.join(result_dir, 'training_log.txt')

# 打开日志文件
with open(log_file_path, 'w',buffering=1) as log_file:
    print("Starting training...")
    log_file.write("Starting training...\n")

    for epoch in range(config.epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, config.device)
        valid_loss, valid_accuracy = validate_model(model, valid_loader, criterion, config.device)
        _, train_accuracy = validate_model(model, train_loader, criterion, config.device)
        


        # 输出并记录每个epoch的结果
        log_message = (f"Epoch {epoch + 1}/{config.epochs}, "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Validation Loss: {valid_loss:.4f}, "
                       f"Train Accuracy: {train_accuracy:.4f}, "
                       f"Validation Accuracy: {valid_accuracy:.4f}\n")
        if valid_accuracy>=0.999 and train_accuracy>=0.999:
            break
        print(log_message.strip())
        log_file.write(log_message)

    print("Training completed!")
    log_file.write("Training completed!\n")

# 保存模型
model_save_path = os.path.join(result_dir, 'model.pth')
torch.save(model, model_save_path)
print(f"Model saved to {model_save_path}")