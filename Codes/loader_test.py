'''
Author: lee12345 15116908166@163.com
Date: 2024-12-16 10:12:28
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-16 15:39:03
FilePath: /Gnn/DHGNN-LSTM/Codes/loader_test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch
import torch.nn as nn
import torch.optim as optim
from src import GraphDataLoader,CombinedModel,SequenceEncoder
from src import Config
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        
ip_encoder = SequenceEncoder(device='cuda:1' if torch.cuda.is_available() else 'cpu',model_path='/home/lzy/Gnn/DHGNN-LSTM/Codes/src/model/')

encoders1 = {
    'ip': ip_encoder  # 将 IP 列的编码器传入
}
encoders2 = {
    'name': ip_encoder  # 将 IP 列的编码器传入
}

path='../datas'

print("Initializing data loader...")
dataLoader1=GraphDataLoader(path,encoders1,encoders2)
dataLoader1.process_data()
train_loader=dataLoader1.get_train_loader()
test_loader=dataLoader1.get_valid_loader()

# 定义超参数
input_dim = 256
hidden_dim = 512
learning_rate = 0.001
epochs = 20

time_deltas = [0, 10, 10]  # 时间差分参数


# 初始化模型
print("Initializing model...")
config=Config()
device=config.device
print(device)

model = CombinedModel(input_dim, hidden_dim)
model.to(device)




# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

optimizer.zero_grad()
model.train()
total_loss = 0.0

for batch in train_loader:
    graphs, ips, labels = batch
    graphs = [graph.to(device) for graph in graphs]
    
    ips = [
        {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in ip_dict.items()}
        for ip_dict in ips
    ]
    
    outputs,global_ips = model(time_deltas,graphs, ips)
    
    labels_list = []
    for ip in global_ips:
        label = labels[ip]
        labels_list.append(label)
    # 将列表转换为一个 torch.tensor
    labels_tensor = torch.tensor(labels_list).to(device)
    
    loss = criterion(outputs, labels_tensor)  # 计算交叉熵损失
    loss.backward()
    optimizer.step()
    print(loss)
    print(torch.softmax(outputs, dim=-1))
    
# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        
        # 获取输入数据
        graphs, labels = batch  
        graphs = graphs.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs, _ = model(time_deltas)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)