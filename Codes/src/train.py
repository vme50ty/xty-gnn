'''
Author: lee12345 15116908166@163.com
Date: 2024-11-20 09:26:58
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-11-22 09:49:53
FilePath: /Gnn/DHGNN-LSTM/Codes/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AEf
'''
import torch
from src import Config
from src import GraphDataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from src import CombinedModel


class TrainPipeline:
    def __init__(self, data_folder, model_class, hidden_size, hidden_channels, config:Config,
                 batch_size=1, lr=0.001, epochs=50, time_intervals=None):
        """
        训练流水线类，用于管理模型训练的所有流程。

        Parameters:
        - data_folder: 数据文件夹路径
        - model_class: 模型类（如 CombinedModel）
        - hidden_size: 模型中时间 LSTM 的隐藏层大小
        - hidden_channels: 模型中 GNN 模块的隐藏通道数
        - batch_size: 数据加载的批次大小
        - lr: 学习率
        - epochs: 训练轮数
        - time_intervals: 图时间间隔列表
        - device: 运行设备
        """
        self.model_class = model_class
        self.hidden_size = hidden_size
        self.hidden_channels = hidden_channels
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.time_intervals = time_intervals or [0.1] * 10  # 默认时间间隔
        self.config=config
        self.device = config.device
        self.data_folder = config.dataPath
        
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None
        
    def setup_data(self):
        """
        加载并处理数据，初始化训练和测试集 DataLoader。
        """
        data_loader = GraphDataLoader(folder_father=self.data_folder, batch_size=self.batch_size)
        data_loader.process_data()
        self.train_loader = data_loader.get_train_loader()
        self.test_loader = data_loader.get_test_loader()
        
    def setup_model(self):
        """
        初始化模型、损失函数和优化器。
        """
        batched_graphs, _ = next(iter(self.train_loader))  # 使用第一个批次初始化模型
        self.model = self.model_class(self.hidden_size, self.hidden_channels, batched_graphs).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
    def train_epoch(self):
        """
        单轮训练。
        """
        self.model.train()
        epoch_loss = 0
        all_preds, all_labels = [], []

        for batched_graphs, labels in self.train_loader:
            batched_graphs = [graph.to(self.device) for graph in batched_graphs]
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            user_probs = self.model(batched_graphs, self.time_intervals)
            loss = self.criterion(user_probs, labels)

            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            all_preds.extend(torch.argmax(user_probs, dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        return epoch_loss / len(self.train_loader), accuracy
    
    def evaluate(self):
        """
        在测试集上验证模型。
        """
        self.model.eval()
        epoch_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batched_graphs, labels in self.test_loader:
                batched_graphs = [graph.to(self.device) for graph in batched_graphs]
                labels = labels.to(self.device)

                user_probs = self.model(batched_graphs, self.time_intervals)
                loss = self.criterion(user_probs, labels)

                epoch_loss += loss.item()
                all_preds.extend(torch.argmax(user_probs, dim=-1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        return epoch_loss / len(self.test_loader), accuracy

    def train(self):
        """
        主训练方法。
        """
        self.setup_data()
        self.setup_model()

        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()

            print(f"Epoch {epoch + 1}/{self.epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        self.save_model()
        
    def save_model(self, path="combined_model.pth"):
        """
        保存模型到指定路径。
        """
        torch.save(self.model.state_dict(), path)
        print(f"模型已保存到 {path}")
        
if __name__ == "__main__":
    # 参数配置
    DATA_FOLDER = "/path/to/data"
    HIDDEN_SIZE = 128
    HIDDEN_CHANNELS = 64
    BATCH_SIZE = 1
    LR = 0.001
    EPOCHS = 50

    # 初始化并运行训练流水线
    pipeline = TrainPipeline(
        data_folder=DATA_FOLDER,
        model_class=CombinedModel,
        hidden_size=HIDDEN_SIZE,
        hidden_channels=HIDDEN_CHANNELS,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS
    )
    pipeline.train()