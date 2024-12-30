'''
Author: lee12345 15116908166@163.com
Date: 2024-11-19 09:41:03
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-25 17:25:08
FilePath: /Gnn/DHGNN-LSTM/Codes/src/CombinedModel.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.typing import Tensor
from src import GnnModel
from src import TimeLSTM,Config
import time
from typing import List,Dict
from collections import defaultdict

class CombinedModel(torch.nn.Module):
    def __init__(self,  hidden_size, hidden_channels):
        super(CombinedModel, self).__init__()
        
        self.config = Config()
        
        # 时间 LSTM 模块
        self.time_lstm = TimeLSTM(hidden_channels, hidden_size)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3)
        )
        
        self.hidden_channels=hidden_channels
        
        self.device=self.config.device
    
        self.gnn_model = GnnModel(self.hidden_channels)
        
        self.to(self.device)
        
    def forward(self,  time_intervals: List[float], batched_graphs: List[HeteroData],ip_mapping: List[Dict[str, int]]):
        """
        :param batched_graphs: 图列表，每个图是一个 HeteroData 对象
        :param time_intervals: 图产生的时间间隔列表，长度应比 batched_graphs 少 1
        :return: 用户类别概率分布
        """
        
        if len(time_intervals) != len(batched_graphs) :
            raise ValueError("time_intervals 的长度%d与 batched_graphs%d不同。",len(time_intervals),len(batched_graphs))

        # Step 1: 获取所有图的用户嵌入
        user_embeddings_all = []  # 存储所有图的用户嵌入
        for data in batched_graphs:
            x_dict = self.gnn_model(data) 
            user_embeddings_all.append(x_dict["user"])  # 收集当前图的用户嵌入

        aligned_embeddings, global_ip_list = self.align_embeddings(user_embeddings_all, ip_mapping)

        # Step 2: 生成时间间隔张量
        time_deltas = torch.tensor(time_intervals, device=self.device).unsqueeze(-1)

        # Step 3: 时间 LSTM 聚合
        time_agg_embeddings, _ = self.time_lstm(aligned_embeddings, time_deltas)
        
        # Step 4: 分类器
        user_logits = self.classifier(time_agg_embeddings)  # [num_users, num_classes]
        
        return user_logits, global_ip_list
    
    def align_embeddings(self, user_embeddings_list, ip_mappings):
        all_ips = set(ip for mapping in ip_mappings for ip in mapping)
        global_ip_list = sorted(all_ips)
        global_ip_map = {ip: idx for idx, ip in enumerate(global_ip_list)}

        num_graphs = len(user_embeddings_list)
        embedding_dim = user_embeddings_list[0].size(1)
        global_user_count = len(global_ip_list)
        aligned_embeddings = torch.zeros((num_graphs, global_user_count, embedding_dim), device=self.device)

        for i, (embeddings, mapping) in enumerate(zip(user_embeddings_list, ip_mappings)):
            for ip, local_idx in mapping.items():
                global_idx = global_ip_map[ip]
                aligned_embeddings[i, global_idx] = embeddings[local_idx]

        return aligned_embeddings, global_ip_list