from src import LoadHeteroGraph
import os
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

class GraphDataLoader:
    def __init__(self,folder_father,encoders1=None, encoders2=None, test_ratio=0.2, batch_size=1):
        """
        图数据加载器类，自动加载和处理文件夹内的图数据。
        Parameters:
        - folder_path: 数据文件夹路径
        - encoders1: 用于处理节点特征的编码器（例如用于 'proxy' 和 'server' 节点）
        - encoders2: 用于处理其他类型节点的编码器（例如用于 'user' 节点）
        - test_ratio: 测试集比例，默认为 20%
        - batch_size: 每批次的图数量
        """
        self.folder_father = folder_father
        self.encoders1=encoders1
        self.encoders2=encoders2
        self.test_ratio=test_ratio
        self.batch_size=batch_size
        self.train_data=[]
        self.train_loader=None
        self.test_loader=None
        
    def load_graph_from_subfolder(self, sub_folder):
        data_loader = LoadHeteroGraph()
        
        for file_name in os.listdir(sub_folder):
            file_path = os.path.join(sub_folder, file_name)
            if "proxys" in file_name:
                data_loader.load_node_csv(file_path, 'id', 'proxy', self.encoders1)
                data_loader.load_edge_csv(file_path,'id','belong','proxy','server','proxy2server')
            elif "servers" in file_name:
                data_loader.load_node_csv(file_path, 'id', 'server', self.encoders1)
            elif "users" in file_name:
                data_loader.load_node_csv(file_path, 'id', 'user', self.encoders2)
                data_loader.load_edge_csv(file_path,'id','belong','user','proxy','user2proxy')
                
        # 添加完全连接边
        data_loader.add_fully_connected_edges(node_type='user')

        # 返回构建的 HeteroData 对象
        return data_loader.get_data()
    
    def load_graphs_from_folder(self, folder_path):
        graphs = []
        for sub_folder in os.listdir(folder_path):
            sub_path = os.path.join(folder_path, sub_folder)
            if os.path.isdir(sub_path):
                graph = self.load_graph_from_subfolder(sub_path)
                graphs.append(graph)
        return graphs 
    
    def process_data(self):
        folders = [os.path.join(self.folder_father, folder) for folder in os.listdir(self.folder_father) if os.path.isdir(os.path.join(self.folder_father, folder))]
        self.train_folders, self.test_folders = train_test_split(folders, test_size=self.test_ratio, random_state=42)
        
        self.train_loader = self.create_loader(self.train_folders)
        self.test_loader = self.create_loader(self.test_folders)
        
    def create_loader(self, folder_list):
        all_data = []
        for folder in folder_list:
            graphs = self.load_graphs_from_folder(folder)
            all_data.append((graphs, self.get_folder_label(folder)))  # 每个 folder 的图集合与标签
        return DataLoader(all_data, batch_size=self.batch_size, shuffle=True)
    
    def get_train_loader(self):
        """
        返回训练集 DataLoader。
        """
        if self.train_loader is None:
            raise ValueError("Train loader is not initialized. Call `process_data()` first.")
        return self.train_loader
    
    def get_test_loader(self):
        """
        返回测试集 DataLoader。
        """
        if self.test_loader is None:
            raise ValueError("Test loader is not initialized. Call `process_data()` first.")
        return self.test_loader
    
    def get_folder_label(self, folder_path):
        pass