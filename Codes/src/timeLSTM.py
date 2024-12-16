'''
Author: lee12345 15116908166@163.com
Date: 2024-10-28 16:53:02
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-16 16:06:26
FilePath: /Gnn/DHGNN-LSTM/Codes/src/time-LSTM.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
from src import Config

class TimeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TimeLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.config=Config()
        self.device=self.config.device
        
        # Define LSTM parameters
        self.Wxi = nn.Linear(input_size, hidden_size).to(self.device)
        self.Whi = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.Wxf = nn.Linear(input_size, hidden_size).to(self.device)
        self.Whf = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.Wxt = nn.Linear(input_size, hidden_size).to(self.device)
        self.Wtt = nn.Linear(1, hidden_size).to(self.device)
        self.Wxc = nn.Linear(input_size, hidden_size).to(self.device)
        self.Whc = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.Wxo = nn.Linear(input_size, hidden_size).to(self.device)
        self.Wto = nn.Linear(1, hidden_size).to(self.device)
        self.Who = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.Wco = nn.Linear(hidden_size, hidden_size).to(self.device)


    # h_prev:上一个时间步的隐藏状态
    def forward(self, x, delta_t, h_prev, c_prev):
        # 输入门
        h_prev=h_prev.to(self.device)
        c_prev=c_prev.to(self.device)
        i_t = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev))
        
        # 遗忘门
        f_t = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev))
        
        delta_t = delta_t.clone().detach().float().view(-1, 1).to(self.device)
        
        delta_t_transformed = self.Wtt(delta_t)
        T_t = torch.sigmoid(self.Wxt(x) + torch.sigmoid(delta_t_transformed))

        # 候选细胞状态
        C_hat_t = torch.tanh(self.Wxc(x) + self.Whc(h_prev))

        # 新的细胞状态
        c_t = f_t * c_prev + i_t * T_t * C_hat_t

        # Output gate and hidden state
        o_t = torch.sigmoid(self.Wxo(x) + self.Wto(delta_t) + self.Who(h_prev) + self.Wco(c_t))
        
        # 当前时间步的隐藏状态
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cell = TimeLSTMCell(input_size, hidden_size)

    def forward(self, node_embeddings, time_deltas):
        """
        :param node_embeddings: [num_graphs, num_users, embedding_dim]
        :param time_deltas: [num_graphs - 1]，图之间的时间间隔
        """
        num_graphs, num_users, _ = node_embeddings.size()
        # 初始化隐藏状态和记忆单元
        h = torch.zeros(num_users, self.hidden_size)  # [num_users, hidden_size]
        c = torch.zeros(num_users, self.hidden_size)  # [num_users, hidden_size]
        
        # 存储所有时间步的隐藏状态
        hidden_states = []
        
        # 循环处理每个图
        for i in range(num_graphs):
            x = node_embeddings[i]  # 当前图的用户嵌入 [num_users, embedding_dim]
            delta_t = time_deltas[i]   # 当前时间间隔
            
            # 更新隐藏状态和记忆单元
            h, c = self.cell(x, delta_t, h, c)  # 支持批处理
            hidden_states.append(h)  # 添加当前时间步的隐藏状态
        
        # 计算累积时间间隔
        cumulative_time_deltas = torch.cumsum(time_deltas, dim=0)  

        # 计算时间权重
        weights = torch.softmax(-cumulative_time_deltas.float(), dim=0)  # 时间权重与间隔反相关
        weights = weights.view(-1, 1)  # 转换为 [num_graphs, 1]，方便广播与hidden_states相乘
        
        # 按权重聚合所有时间步的隐藏状态
        hidden_states = torch.stack(hidden_states, dim=0)  # [num_graphs, num_users, hidden_size]
        time_agg_embedding = torch.sum(hidden_states * weights.unsqueeze(-1), dim=0)  # [num_users, hidden_size]
        
        # 返回聚合嵌入和所有隐藏状态
        return time_agg_embedding, hidden_states