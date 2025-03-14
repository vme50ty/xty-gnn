'''
Author: lee12345 15116908166@163.com
Date: 2024-11-20 09:45:23
LastEditors: vme50ty 15116908166@163.com
LastEditTime: 2025-03-12 23:58:25
FilePath: /Gnn/DHGNN-LSTM/Codes/src/dataLoader.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from src import LoadHeteroGraph,Config
import os
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np

class GraphDataLoader:
    def __init__(self,folder_father,encoders1=None, encoders2=None,  batch_size=1):
        """
        图数据加载器类，自动加载和处理文件夹内的图数据。
        Parameters:
        - folder_path: 数据文件夹路径
        - encoders1: 用于处理节点特征的编码器（例如用于 'proxy' 和 'server' 节点）
        - encoders2: 用于处理其他类型节点的编码器（例如用于 'user' 节点）
        - test_ratio: 测试集比例，默认为 20%
        - batch_size: 每批次的图数量
        """
        self.config=Config()
        self.folder_father = folder_father
        self.encoders1=encoders1
        self.encoders2=encoders2
        self.test_ratio=self.config.test_ratio
        self.batch_size=self.config.batch_size
        self.train_data=[]
        self.train_loader=None
        self.test_loader=None
        
    def load_graph_from_subfolder(self, sub_folder): 
        data_loader = LoadHeteroGraph()
        user_ip_to_index = {}
        
        for file_name in os.listdir(sub_folder):
            file_path = os.path.join(sub_folder, file_name)
            if "proxys" in file_name:
                proxy_mapping=data_loader.load_node_csv(file_path, 'id', 'proxy', None,['proxyname','belong'])
            elif "users" in file_name:
                # print(file_path)
                user_mapping = data_loader.load_node_csv(file_path, 'userIP', 'user', None,['id'])
                for user_ip, index in user_mapping.items():
                    user_ip_to_index[user_ip] = index
                data_loader.load_edge_csv(file_path,'userIP','belong','user','proxy','user2proxy')
                
        # 添加完全连接边
        # data_loader.add_fully_connected_edges(node_type='user')
        
        data = data_loader.get_data()
        # # 打印 proxy 节点特征
        # print("Proxy Node Features:")
        # print(data['proxy'].x) 
        # # 打印 user 节点特征
        # print("User Node Features:")
        # print(data['user'].x)  # 输出 [250, 6] 的矩阵，注意如果没有特征会是 None
        # # 打印 user -> proxy 边
        # print("User to Proxy Edges (edge_index):")
        # print(data['user', 'user2proxy', 'proxy'].edge_index)
        # # 打印 proxy -> user 边（反向边）
        # print("Proxy to User Edges (edge_index):")
        # print(data['proxy', 'rev_user2proxy', 'user'].edge_index)
        # print("User ip to index:")
        # print(user_ip_to_index)
        # print("proxy to index:")
        # print(proxy_mapping)
        
        return data, user_ip_to_index
    
    def load_graphs_from_folder(self, folder_path):
        graphs = []
        ip_mappings = []
        sub_folders = sorted([f for f in os.listdir(folder_path) if f.isdigit()], key=int)  # 确保按时间排序
        time_delays = [5]  # 初始化，第一个时间间隔固定为 5
        if len(sub_folders) > 1:
            # 计算时间间隔：当前文件夹时间戳 - 上一个文件夹时间戳
            timestamps = [int(folder) for folder in sub_folders]  # 转换为整数时间戳
            time_diffs = np.diff(timestamps).tolist()  # 计算相邻文件夹时间差
            time_delays.extend(time_diffs)  # 追加时间间隔
        for sub_folder in sub_folders:
            sub_path = os.path.join(folder_path, sub_folder)
            if os.path.isdir(sub_path):
                graph, ip_mapping = self.load_graph_from_subfolder(sub_path)
                # print(sub_path)
                # print(graph)
                # print(ip_mapping)
                graphs.append(graph)
                ip_mappings.append(ip_mapping)
        return graphs , ip_mappings ,time_delays
    
    def process_data(self):
        folders = [os.path.join(self.folder_father, folder) 
                   for folder in os.listdir(self.folder_father) 
                   if os.path.isdir(os.path.join(self.folder_father, folder))]
        self.train_folders, self.test_folders = train_test_split(
            folders, test_size=self.test_ratio, random_state=42
        )
        self.train_loader = self.create_loader(self.train_folders)
        self.test_loader = self.create_loader(self.test_folders)
        
    def create_loader(self, folder_list):
        all_data = []
        for folder in folder_list:
            graphs, ip_mappings,timedelates = self.load_graphs_from_folder(folder)  # 获取 graphs 和 ip_mappings
            labels = self.get_folder_label(folder)  # 获取当前 folder 的标签,在每个folder文件夹下，存在label.csv
            
            # 将 graphs 和 ip_mappings 一起存储
            all_data.append((graphs, ip_mappings, labels,timedelates))
        
        # DataLoader 中返回完整的数据
        return DataLoader(all_data, batch_size=self.batch_size, shuffle=True)
    
    def get_train_loader(self):
        """
        返回训练集 DataLoader。
        """
        if self.train_loader is None:
            raise ValueError("Train loader is not initialized. Call `process_data()` first.")
        return self.train_loader
    
    def get_valid_loader(self):
        """
        返回测试集 DataLoader。
        """
        if self.test_loader is None:
            raise ValueError("Test loader is not initialized. Call `process_data()` first.")
        return self.test_loader

    def get_folder_label(self, folder_path):

        label_file = os.path.join(folder_path, "label.csv")
        
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"Label file not found in folder: {folder_path}")
        
        # 读取 CSV 文件
        df = pd.read_csv(label_file)
        
        # 确保文件格式正确
        if not {"id", "normal", "direct", "slow",'steal'}.issubset(df.columns):
            raise ValueError(f"Label file {label_file} has incorrect format. Expected columns: 'id', 'normal', 'direct', 'slow','steal'")
        
        # 将标签转化为数值
        labels = {}
        for _, row in df.iterrows():
            user_ip = row["id"]
            if row["normal"] == 1:
                labels[user_ip] = 0  # normal
            elif row["direct"] == 1:
                labels[user_ip] = 1  # abnormal
            elif row["slow"] == 1:
                labels[user_ip] = 2  # unknown
            elif row["steal"] == 1:
                labels[user_ip] = 3
            else:
                raise ValueError(f"Invalid label in row: {row}")
        
        return labels