'''
Author: lee12345 15116908166@163.com
Date: 2024-10-29 10:52:29
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-10-31 15:35:59
FilePath: /Gnn/DHGNN-LSTM/Codes/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from src import LoadHeteroGraph
from src import Config
from src import SequenceEncoder
from src import GnnModel
import torch

# print("CUDA Available:", torch.cuda.is_available())
# print("CUDA Device Count:", torch.cuda.device_count())

# if torch.cuda.is_available():
#     for i in range(torch.cuda.device_count()):
#         print(f"Device {i}: {torch.cuda.get_device_name(i)}")

ip_encoder = SequenceEncoder(device='cuda:1' if torch.cuda.is_available() else 'cpu',model_path='/home/lzy/Gnn/DHGNN-LSTM/Codes/src/model/')

# 定义编码器参数
encoders1 = {
    'ip': ip_encoder  # 将 IP 列的编码器传入
}
encoders2 = {
    'name': ip_encoder  # 将 IP 列的编码器传入
}
dataLoad = LoadHeteroGraph()
path_proxy='./data_folder_20241029_110258/proxys.csv'
path_user='./data_folder_20241029_110258/users.csv'
path_server='./data_folder_20241029_110258/servers.csv'
dataLoad.load_node_csv(path_proxy,'id','proxy',encoders1)
dataLoad.load_node_csv(path_server,'id','server',encoders1)
dataLoad.load_node_csv(path_user,'id','user',encoders2)
dataLoad.load_edge_csv(path_user,'id','belong','user','proxy','user2proxy')
dataLoad.load_edge_csv(path_proxy,'id','belong','proxy','server','proxy2server')

dataLoad.add_fully_connected_edges(node_type='user')
# print(hasattr(dataLoad.data, 'metadata'))  # 检查是否有 metadata() 方法
Model = GnnModel(512,dataLoad.data)

# 前向传播获取嵌入
with torch.no_grad():  # 关闭梯度计算以加速推理
    embeddings = Model(dataLoad.data)
    
print(f'embeddings:{embeddings}')
# print(dataLoad.data)
#  for node_type in dataLoad.data.node_types:
#      print(f"Node type: {node_type}")
#      print(f"Features:\n{dataLoad.data[node_type].x}\n")

# # 查看每种边类型的边索引
# for edge_type in dataLoad.data.edge_types:
#     print(f"Edge type: {edge_type}")
#     print(f"Edge index:\n{dataLoad.data[edge_type].edge_index}\n")




