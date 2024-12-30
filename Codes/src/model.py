'''
Author: lee12345 15116908166@163.com
Date: 2024-10-28 09:50:58
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-30 16:04:08
FilePath: /Gnn/DHGNN-LSTM/Codes/src/model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torch_geometric.nn import to_hetero, MessagePassing
import torch.nn.functional as F
from torch import nn, Tensor
import torch
from torch_geometric.data import HeteroData
from torch_scatter import scatter_softmax, scatter_add
from src import Config
from torch_geometric.nn import HANConv

class WeightParameter(nn.Module):
    def __init__(self, hidden_channels):
        super(WeightParameter, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        nn.init.xavier_uniform_(self.weight)

class HeteroAttentionLayer(MessagePassing):
    def __init__(self, hidden_channels, metadata):
        super(HeteroAttentionLayer, self).__init__()  
        self.hidden_channels = hidden_channels
        self.config=Config()
        self.device=self.config.device
        
        # 为每种节点类型创建独立的查询、键和值向量生成器
        self.q_dict = nn.ModuleDict()
        self.k_dict = nn.ModuleDict()
        self.v_dict = nn.ModuleDict()
        for node_type in metadata[0]:  # metadata[0]包含节点类型
            self.q_dict[node_type] = nn.Linear(hidden_channels, hidden_channels)
            self.k_dict[node_type] = nn.Linear(hidden_channels, hidden_channels)
            self.v_dict[node_type] = nn.Linear(hidden_channels, hidden_channels)

        # 为每种边类型创建独立的权重矩阵
        self.w_dict = nn.ModuleDict()
        for edge_type in metadata[1]:  # metadata[1]包含边类型
            edge_type_str = f"{edge_type}"  # 将元组转换为字符串
            self.w_dict[edge_type_str] = WeightParameter(hidden_channels)  # 使用自定义模块
            
    def forward(self, x_dict, edge_index_dict):
        out_dict = {}
        # print("x_dict=")
        # print(x_dict)
        # print("edge_index_dict=")
        # print(edge_index_dict)

        for node_type, node_feats in x_dict.items():
            if node_type not in out_dict:
                out_dict[node_type] = torch.zeros_like(node_feats)
            
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                if dst_type != node_type:
                    continue  # Skip if the edge's destination does not match the current node type
                
                # Apply linear transformations
                q = self.q_dict[src_type](x_dict[src_type]) # (num_src_nodes, hidden_dim)
                
                k = self.k_dict[dst_type](x_dict[dst_type])# (num_dst_nodes, hidden_dim)
                
                v = self.v_dict[src_type](x_dict[src_type]) # (num_src_nodes, hidden_dim)

                # Extract source and destination nodes for edges
                src_nodes = edge_index[0]  # Source nodes for edges
                dst_nodes = edge_index[1]

                # Compute raw attention scores (alpha_uv)
                W = self.w_dict[f"{edge_type}"].weight  # Weight matrix for this edge type
                
                raw_alpha = (q[src_nodes] @ W @ (k[dst_nodes].T)).sum(dim=-1) / (self.hidden_channels ** 0.5)  # (num_edges,)
                # Normalize attention scores for each destination node independently
                # print(raw_alpha)
                normalized_alpha = scatter_softmax(raw_alpha, dst_nodes, dim=0)  # (num_edges,)
                # print(normalized_alpha)
                
                # Compute weighted messages
                weighted_messages = normalized_alpha.unsqueeze(-1) * v[src_nodes]  # (num_edges, hidden_dim)
                
                # Aggregate messages to destination nodes
                out_dict[dst_type] = scatter_add(
                    weighted_messages, dst_nodes, dim=0, out=out_dict[dst_type]
                )
                

        return out_dict

class HANLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, metadata):
        super(HANLayer, self).__init__()
        self.han_conv = HANConv(in_channels, out_channels, metadata)

    def forward(self, x_dict, edge_index_dict):
        return self.han_conv(x_dict, edge_index_dict)

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.conv1 = HeteroAttentionLayer(hidden_channels, metadata)
        self.conv2 = HeteroAttentionLayer(hidden_channels, metadata)
        
        # 自环信息线性变换
        self.self_loop_transform = nn.ModuleDict({
            node_type: nn.Linear(hidden_channels, hidden_channels) 
            for node_type in metadata[0]
        })
        for transform in self.self_loop_transform.values():
            nn.init.xavier_uniform_(transform.weight)  # Xavier 初始化

        # 归一化层
        self.batch_norms = nn.ModuleDict({
            node_type: nn.BatchNorm1d(hidden_channels) for node_type in metadata[0]
        })

    def forward(self, x_dict, edge_index_dict):
        original_x_dict = {node_type: x.clone() for node_type, x in x_dict.items()}
        
        # 第一层异构图卷积
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {node_type: F.relu(self.batch_norms[node_type](x)) for node_type, x in x_dict.items()} 
        
        # 自环信息融合
        for node_type, x in x_dict.items():
            self_loop_info = self.self_loop_transform[node_type](original_x_dict[node_type])
            x_dict[node_type] = x + self_loop_info  # 融合自环信息
        
        # 第二层异构图卷积
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        # 自环信息融合
        for node_type, x in x_dict.items():
            self_loop_info = self.self_loop_transform[node_type](original_x_dict[node_type])
            x_dict[node_type] = x + self_loop_info  # 融合自环信息
            
        return x_dict
    

class GnnModel(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        # 使用 ModuleDict 动态管理每种节点类型的线性层
        self.lin_layers = torch.nn.ModuleDict()
        self.gnn = None  # 将 GNN 延迟初始化

    def _initialize_layer(self, data: HeteroData, node_type: str):
        feat_dim = data[node_type].x.shape[1]
        if node_type not in self.lin_layers:
            # 初始化线性层
            self.lin_layers[node_type] = torch.nn.Linear(feat_dim, self.hidden_channels)

    def forward(self, data: HeteroData) -> dict:
        # 动态初始化线性层
        for node_type in data.node_types:
            self._initialize_layer(data, node_type)

        # 延迟初始化 GNN 模型
        if self.gnn is None:
            self.gnn = GNN(self.hidden_channels, data.metadata())
        device = data[data.node_types[0]].x.device  
        self.to(device)  # 将模型迁移到数据所在设备

        # 准备输入特征
        x_dict = {}
        for node_type in data.node_types:
            x_dict[node_type] = self.lin_layers[node_type](data[node_type].x)

        return self.gnn(x_dict, data.edge_index_dict)