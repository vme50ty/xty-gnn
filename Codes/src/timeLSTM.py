'''
Author: lee12345 15116908166@163.com
Date: 2024-10-28 16:53:02
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-10-28 17:04:01
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

    def forward(self, x, delta_t, h_prev, c_prev):
        # Input, Forget, and Temporal gates
        i_t = torch.sigmoid(self.Wxi(x) + self.Whi(h_prev))
        f_t = torch.sigmoid(self.Wxf(x) + self.Whf(h_prev))
        T_t = torch.sigmoid(self.Wxt(x) + torch.sigmoid(delta_t * self.Wtt))

        # Candidate cell state
        C_hat_t = torch.tanh(self.Wxc(x) + self.Whc(h_prev))

        # Update cell state with temporal gate T_t
        c_t = f_t * c_prev + i_t * T_t * C_hat_t

        # Output gate and hidden state
        o_t = torch.sigmoid(self.Wxo(x) + delta_t * self.Wto + self.Who(h_prev) + self.Wco(c_t))
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

        # Output the last hidden state as the time-aggregated embedding
        return hidden_states[-1], hidden_states