'''
Author: lee12345 15116908166@163.com
Date: 2024-10-28 16:53:02
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-11-19 10:56:03
FilePath: /Gnn/DHGNN-LSTM/Codes/src/time-LSTM.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn

class TimeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TimeLSTMCell, self).__init__()
        self.hidden_size = hidden_size

        # Define LSTM parameters
        self.Wxi = nn.Linear(input_size, hidden_size)
        self.Whi = nn.Linear(hidden_size, hidden_size)
        self.Wxf = nn.Linear(input_size, hidden_size)
        self.Whf = nn.Linear(hidden_size, hidden_size)
        self.Wxt = nn.Linear(input_size, hidden_size)
        self.Wtt = nn.Linear(1, hidden_size)
        self.Wxc = nn.Linear(input_size, hidden_size)
        self.Whc = nn.Linear(hidden_size, hidden_size)
        self.Wxo = nn.Linear(input_size, hidden_size)
        self.Wto = nn.Linear(1, hidden_size)
        self.Who = nn.Linear(hidden_size, hidden_size)
        self.Wco = nn.Linear(hidden_size, hidden_size)

    # h_prev:上一个时间步的隐藏状态
    def forward(self, x, delta_t, h_prev, c_prev):
        # 输入门
        i_t = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev))
        
        # 遗忘门
        f_t = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev)) 
        
        delta_t = torch.tensor(delta_t).float().view(-1, 1).to(x.device)
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
        h, c = (torch.zeros(self.hidden_size), torch.zeros(self.hidden_size))
        hidden_states = []
        
        for x, delta_t in zip(node_embeddings, time_deltas):
            h, c = self.cell(x, delta_t, h, c)
            hidden_states.append(h)
        
        cumulative_time_deltas = torch.cumsum(torch.tensor(time_deltas), dim=0)
        weights = torch.softmax(-cumulative_time_deltas.float(), dim=0)  # 权重与时间反相关
        time_agg_embedding = torch.sum(torch.stack(hidden_states) * weights.unsqueeze(-1), dim=0)
        
        # Output the last hidden state as the time-aggregated embedding
        return time_agg_embedding, hidden_states