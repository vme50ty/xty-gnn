'''
Author: lee12345 15116908166@163.com
Date: 2024-11-19 09:41:03
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-11-20 09:30:10
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

class CombinedModel(torch.nn.Module):
    def __init__(self,  hidden_size, hidden_channels, data: HeteroData):
        super(CombinedModel, self).__init__()
        
        self.config = Config()
        # 图神经网络模块
        self.gnn_model = GnnModel(hidden_channels, data)
        
        # 时间 LSTM 模块
        self.time_lstm = TimeLSTM(hidden_channels, hidden_size)
        
        # 分类器
        self.classifier = nn.Linear(hidden_size, 3)  # 输出 3 类（正常用户/异常用户/未知用户）

    def forward(self,data:HeteroData,user_time_sequences,time_deltas,lastTime):
        # Step 1: GNN 模块计算当前用户嵌入
        x_dict=self.gnn_model(data)
        user_embedings=x_dict["user"]
        
        # Step 2: 更新时间序列嵌入
        if len(user_time_sequences)<self.config.graph_nums:
            user_time_sequences.append(user_embedings)
            delta_t = 0 if len(user_time_sequences) == 1 else time.time() - lastTime
            time_deltas.append(delta_t)
               
        else :
            user_time_sequences.pop(0)  # 去掉最久的嵌入
            time_deltas.pop(0)
            
            user_time_sequences.append(user_embedings)
            time_deltas.append(time.time() - lastTime)
            
        # Step 3: 时间 LSTM 生成聚合嵌入
        # 时间 LSTM 聚合
        time_agg_embeddings, _ = self.time_lstm(
            torch.stack(user_time_sequences),
            torch.tensor(time_deltas).unsqueeze(-1)
        )
        
        user_logits = self.classifier(time_agg_embeddings)
        user_probs = torch.softmax(user_logits, dim=-1)  # 转为概率分布
        return user_probs,user_time_sequences,time_deltas