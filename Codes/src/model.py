'''
Author: lee12345 15116908166@163.com
Date: 2024-10-28 09:50:58
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-11-19 10:46:49
FilePath: /Gnn/DHGNN-LSTM/Codes/src/model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from torch_geometric.nn import to_hetero, MessagePassing
import torch.nn.functional as F
from torch import nn, Tensor
import torch
from torch_geometric.data import HeteroData
from torch_scatter import scatter_softmax, scatter_add

class WeightParameter(nn.Module):
    def __init__(self, hidden_channels):
        super(WeightParameter, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
        nn.init.xavier_uniform_(self.weight)

class HeteroAttentionLayer(MessagePassing):
    def __init__(self, hidden_channels, metadata):
        super(HeteroAttentionLayer, self).__init__(aggr='add')  # 聚合方法为“加和”
        self.hidden_channels = hidden_channels

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
        
        self.self_attention_dict = nn.ModuleDict()
        for node_type in metadata[0]:  # metadata[0]包含节点类型
            self.self_attention_dict[node_type] = nn.Linear(hidden_channels, hidden_channels)
            
    def forward(self, x_dict, edge_index_dict):
        out_dict = {}

        for node_type, node_feats in x_dict.items():
            if node_type not in out_dict:
                out_dict[node_type] = torch.zeros_like(node_feats)

            q_self = self.q_dict[node_type](x_dict[node_type])  # 自身查询向量
            k_self = self.k_dict[node_type](x_dict[node_type])  # 自身键向量
            v_self = self.v_dict[node_type](x_dict[node_type])  # 自身值向量
            raw_alpha_self = (q_self * k_self).sum(dim=-1) / (self.hidden_channels ** 0.5)  # 自注意力得分
            normalized_alpha_self = F.softmax(raw_alpha_self, dim=0)  # 自注意力归一化
            self_message = (normalized_alpha_self.unsqueeze(-1) * v_self)  # 自注意力消息
            
            for edge_type, edge_index in edge_index_dict.items():
                src_type, _, dst_type = edge_type
                if dst_type != node_type:
                    continue  # Skip if the edge's destination does not match the current node type

                # Apply linear transformations
                q = self.q_dict[src_type](x_dict[src_type])  # (num_src_nodes, hidden_dim)
                k = self.k_dict[dst_type](x_dict[dst_type])  # (num_dst_nodes, hidden_dim)
                v = self.v_dict[src_type](x_dict[src_type])  # (num_src_nodes, hidden_dim)

                # Extract source and destination nodes for edges
                src_nodes = edge_index[0]  # Source nodes for edges
                dst_nodes = edge_index[1]  # Destination nodes for edges

                # Compute raw attention scores (alpha_uv)
                W = self.w_dict[f"{edge_type}"].weight  # Weight matrix for this edge type
                raw_alpha = (q[src_nodes] @ W @ k[dst_nodes].T).sum(dim=-1) / (self.hidden_channels ** 0.5)  # (num_edges,)
                
                # Normalize attention scores for each destination node independently
                normalized_alpha = scatter_softmax(raw_alpha, dst_nodes, dim=0)  # (num_edges,)
                # print(f'normalized_alpha={normalized_alpha}')
                # Compute weighted messages
                weighted_messages = normalized_alpha.unsqueeze(-1) * v[src_nodes]  # (num_edges, hidden_dim)

                # Aggregate messages to destination nodes
                out_dict[dst_type] = scatter_add(
                    weighted_messages, dst_nodes, dim=0, out=out_dict[dst_type]
                )
                out_dict[node_type] += self_message  # 加入自注意力消息

        return out_dict



class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.conv1 = HeteroAttentionLayer(hidden_channels, metadata)
        self.conv2 = HeteroAttentionLayer(hidden_channels, metadata)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

class GnnModel(torch.nn.Module):
    def __init__(self, hidden_channels,data:HeteroData):
        super().__init__()
        # 获取每种节点的特征维度
        proxy_feat_dim = data["proxy"].x.shape[1]
        user_feat_dim = data["user"].x.shape[1]
        server_feat_dim = data["server"].x.shape[1]
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.proxy_lin = torch.nn.Linear(proxy_feat_dim, hidden_channels)
        self.user_lin = torch.nn.Linear(user_feat_dim, hidden_channels)
        self.server_lin=torch.nn.Linear(server_feat_dim, hidden_channels)
        
        self.prxoy_emb = torch.nn.Embedding(data["proxy"].num_nodes, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.server_emb = torch.nn.Embedding(data["server"].num_nodes, hidden_channels)
        
        self.proxy_ids = torch.arange(data["proxy"].num_nodes)
        self.user_ids = torch.arange(data["user"].num_nodes)
        self.server_ids = torch.arange(data["server"].num_nodes)

        # print(data["proxy"])
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels,data.metadata())
        # Convert GNN model into a heterogeneous variant:
        # self.gnn = to_hetero(self.gnn, metadata=data.metadata())     
        
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "proxy": self.proxy_lin(data["proxy"].x) + self.prxoy_emb(self.proxy_ids),
          "user": self.user_lin (data["user"].x) + self.user_emb(self.user_ids),
          "server": self.server_lin(data["server"].x) + self.server_emb(self.server_ids)
        } 
        
        # print(f'first{data.edge_index_dict}')
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        return x_dict