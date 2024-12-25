'''
Author: lee12345 15116908166@163.com
Date: 2024-10-29 10:52:29
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-25 13:58:09
FilePath: /Gnn/DHGNN-LSTM/Codes/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from src import LoadHeteroGraph,GraphDataLoader
from src import Config
from src import SequenceEncoder
from src import GnnModel
from src import CombinedModel

print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

ip_encoder = SequenceEncoder(model_path='/home/lzy/Gnn/DHGNN-LSTM/Codes/src/model/')

# 定义编码器参数
encoders1 = {
    'ip': ip_encoder  # 将 IP 列的编码器传入
}
encoders2 = {
    'name': ip_encoder  # 将 IP 列的编码器传入
}

path='./data_folder'
dataLoader1=GraphDataLoader(path,encoders1,encoders2)
data,ipmapping=dataLoader1.load_graphs_from_folder(path)

time_deltas = [5,10,10,10,10,10,10,10,10]

SModel=CombinedModel(256,512)

# 前向传播获取嵌入

with torch.no_grad():  # 关闭梯度计算以加速推理
    embeddings,ips= SModel(time_deltas,data,ipmapping)
    
print(f'embeddings:{embeddings}')
print(f'ips:{ips}')
# print(dataLoad.data)
#  for node_type in dataLoad.data.node_types:
#      print(f"Node type: {node_type}")
#      print(f"Features:\n{dataLoad.data[node_type].x}\n")

# # 查看每种边类型的边索引
# for edge_type in dataLoad.data.edge_types:
#     print(f"Edge type: {edge_type}")
#     print(f"Edge index:\n{dataLoad.data[edge_type].edge_index}\n")




