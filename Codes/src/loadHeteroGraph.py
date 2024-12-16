'''
Author: lee12345 15116908166@163.com
Date: 2024-10-28 10:11:18
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-16 19:46:36
FilePath: /Gnn/DHGNN-LSTM/Codes/src/make_graph.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import itertools
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer
import torch_geometric.transforms as T
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from src import Config
class LoadHeteroGraph:
    def __init__(self):
        self.data = HeteroData()  # Initialize the heterogeneous graph data
        self.node_mappings = {}  # Store mappings for each type of node

    def load_node_csv(self, path, index_col, node_type, encoders=None, **kwargs):
        """
        从CSV文件加载节点数据并添加到图中。

        参数：
        - path: 节点CSV文件的路径
        - index_col: 包含节点ID的列名
        - node_type: 节点的类型（例如 'user', 'proxy', 'server'）
        - encoders: 一个字典，将列名映射到对应的编码函数
        - **kwargs: 其他传递给`pd.read_csv`的参数
        """
        # 加载节点数据（从CSV文件读取数据，index_col用于指定节点ID列）
        df = pd.read_csv(path, index_col=index_col, **kwargs)
        # 创建一个从节点ID到连续索引的映射
        # self.node_mappings[node_type] 保存的是该类型节点的ID到索引的映射
        self.node_mappings[node_type] = {index: i for i, index in enumerate(df.index.unique())}

        # Handle feature encoding or direct assignment
        if encoders is not None:
            # Apply encoders and concatenate features
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            non_encoded_cols = [col for col in df.columns if col not in encoders]
            if non_encoded_cols:
                non_encoded_features = torch.tensor(df[non_encoded_cols].values, dtype=torch.float)
                xs.append(non_encoded_features)
            # 拼接上述xs和x
            x = torch.cat(xs, dim=-1)
        else:
            # Use raw data as features when no encoder is provided
            x = torch.tensor(df.values, dtype=torch.float)

        self.data[node_type].x = x  # 将特征添加到HeteroData对象中（假设self.data是一个HeteroData对象）
        # print(f'nodetype='+node_type)
        # print(self.node_mappings[node_type])
        return self.node_mappings[node_type]

    def load_edge_csv(self, path, src_index_col, dst_index_col, src_type, dst_type, relation, encoders=None, **kwargs):
        """
        Load edge data from a CSV file and add it to the graph.

        Parameters:
        - path: Path to the edge CSV file
        - src_index_col: Column name for source node IDs
        - dst_index_col: Column name for destination node IDs
        - src_type: Source node type (e.g., 'user')
        - dst_type: Destination node type (e.g., 'proxy')
        - relation: Relationship type (e.g., 'belong')
        - encoders: Dictionary mapping column names to encoder functions
        """
        # Load edge data
        df = pd.read_csv(path, **kwargs)
        
        # Use mappings to get the indices for source and destination nodes
        src = [self.node_mappings[src_type][index] for index in df[src_index_col]]
        dst = [self.node_mappings[dst_type][index] for index in df[dst_index_col]]
        
        # Add edge indices to HeteroData
        self.data[(src_type, relation, dst_type)].edge_index = torch.tensor([src, dst])

        # Apply encoders to edge attributes if provided
        if encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)
            self.data[(src_type, relation, dst_type)].edge_attr = edge_attr
            
    def add_fully_connected_edges(self, node_type, relation="connected"):
        """
        Add edges between all nodes of the specified type to create a fully connected subgraph.
        
        Parameters:
        - node_type: Type of node (e.g., 'user')
        - relation: Relationship type for the edges (default is 'connected')
        """
        # Ensure the node type exists in node_mappings
        if node_type not in self.node_mappings:
            raise ValueError(f"Node type '{node_type}' not found in node_mappings")

        # Get all node indices of the specified type
        node_indices = list(self.node_mappings[node_type].values())

        # Generate fully connected edge pairs
        src, dst = zip(*itertools.combinations(node_indices, 2))  # combinations for undirected edges

        # Convert to torch tensors and add to the HeteroData object
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        self.data[(node_type, relation, node_type)].edge_index = edge_index
        
    def get_data(self):
        """Return the constructed HeteroData object."""
        data1 = T.ToUndirected()(self.data)
        return data1

    

class SequenceEncoder:
    def __init__(self, model_path='./model'):
        self.config=Config()
        self.device =self.config.device
        self.model = SentenceTransformer(model_path, device=self.device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()
    
    