from torch_geometric.nn import to_hetero, MessagePassing
import torch.nn.functional as F
from torch import nn, Tensor
import torch
from torch_geometric.data import HeteroData

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
            self.w_dict[edge_type] = nn.Parameter(torch.Tensor(hidden_channels, hidden_channels))
            nn.init.xavier_uniform_(self.w_dict[edge_type])  # Xavier初始化

    def forward(self, x_dict, edge_index_dict):
        # 使用PyTorch Geometric的MessagePassing机制
        out_dict = {}
        for edge_type, edge_index in edge_index_dict.items():
            src_type, _, dst_type = edge_type

            # 获取源和目标节点的嵌入
            q = self.q_dict[src_type](x_dict[src_type])
            k = self.k_dict[dst_type](x_dict[dst_type])
            v = self.v_dict[src_type](x_dict[src_type])

            # 根据边类型应用不同的权重矩阵
            W = self.w_dict[edge_type]
            alpha = (q[edge_index[0]] @ W @ k[edge_index[1]].T) / (self.hidden_channels ** 0.5)
            alpha = F.leaky_relu(alpha)
            alpha = torch.softmax(alpha, dim=-1)

            # 加权聚合邻居节点信息
            out = alpha.unsqueeze(-1) * v[edge_index[0]]
            out_dict[dst_type] = out_dict.get(dst_type, 0) + torch.scatter_add(out, edge_index[1], dim=0)

        return out_dict


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.conv1 = HeteroAttentionLayer(hidden_channels, metadata)
        self.conv2 = HeteroAttentionLayer(hidden_channels, metadata)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict

class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_movie: Tensor, edge_label_index: Tensor) -> Tensor:
        # 将节点嵌入转换为边表示：
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_movie = x_movie[edge_label_index[1]]
        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_user * edge_feat_movie).sum(dim=-1)
    
class Model(torch.nn.Module):
    def __init__(self, hidden_channels,data):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn two
        # embedding matrices for users and movies:
        self.proxy_lin = torch.nn.Linear(20, hidden_channels)
        self.user_lin = torch.nn.Linear(20, hidden_channels)
        self.server_lin=torch.nn.Linear(20, hidden_channels)
        
        self.prxoy_emb = torch.nn.Embedding(data["proxy"].num_nodes, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.server_emb = torch.nn.Embedding(data["server"].num_nodes, hidden_channels)
        
        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)
        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())     
        self.classifier = Classifier()
        
    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "proxy": self.proxy_lin(data["proxy"].x) + self.movie_emb(data["proxy"].node_id),
          "user": self.proxy_lin(data["user"].x) + self.movie_emb(data["user"].node_id),
          "server": self.proxy_lin(data["server"].x) + self.movie_emb(data["server"].node_id)
        } 
        
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        
        
        return x_dict