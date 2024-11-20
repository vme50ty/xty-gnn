'''
Author: lee12345 15116908166@163.com
Date: 2024-11-19 09:41:03
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-11-20 15:38:38
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
from typing import List

class CombinedModel(torch.nn.Module):
    def __init__(self,  hidden_size, hidden_channels, batched_graphs: List[HeteroData]):
        super(CombinedModel, self).__init__()
        
        self.config = Config()
        # 图神经网络模块
        self.gnn_model = GnnModel(hidden_channels, batched_graphs[0])
        
        # 时间 LSTM 模块
        self.time_lstm = TimeLSTM(hidden_channels, hidden_size)
        
        # 分类器
        self.classifier = nn.Linear(hidden_size, 3)  # 输出 3 类（正常用户/异常用户/未知用户）
        
    def forward(self, batched_graphs: List[HeteroData], time_intervals: List[float]):
        """
        :param batched_graphs: 图列表，每个图是一个 HeteroData 对象
        :param time_intervals: 图产生的时间间隔列表，长度应比 batched_graphs 少 1
        :return: 用户类别概率分布
        """
        if len(time_intervals) != len(batched_graphs) :
            raise ValueError("time_intervals 的长度与 batched_graphs不同。")

        # Step 1: 获取所有图的用户嵌入
        user_embeddings_all = []  # 存储所有图的用户嵌入
        for data in batched_graphs:
            x_dict = self.gnn_model(data)  # 针对每个图运行 GNN 模型
            user_embeddings_all.append(x_dict["user"])  # 收集当前图的用户嵌入

        # 将用户嵌入堆叠为时间序列
        user_embeddings_combined = torch.stack(user_embeddings_all, dim=0)  # [num_graphs, num_users, embedding_dim]

        # Step 2: 生成时间间隔张量
        time_deltas = torch.tensor(time_intervals).unsqueeze(-1)

        # Step 3: 时间 LSTM 聚合
        time_agg_embeddings, _ = self.time_lstm(
            user_embeddings_combined,  # 时间序列的用户嵌入 [num_graphs, num_users, embedding_dim]
            time_deltas  # 时间间隔 [num_graphs-1, num_users]
        )
        print(f'time_agg_embeddings={time_agg_embeddings}')

        # Step 4: 分类器
        user_logits = self.classifier(time_agg_embeddings)  # [num_users, num_classes]
        user_probs = torch.softmax(user_logits, dim=-1)  # 转为概率分布
        
        return user_probs