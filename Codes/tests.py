'''
Author: lee12345 15116908166@163.com
Date: 2024-11-19 10:20:47
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-11-19 10:20:52
FilePath: /Gnn/DHGNN-LSTM/Codes/tests.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from src import TimeLSTMCell
x = torch.randn(1, 16)  # 输入大小为 16
delta_t = 2.0  # 时间间隔
h = torch.zeros(1, 32)  # 隐藏状态
c = torch.zeros(1, 32)  # 记忆单元

cell = TimeLSTMCell(input_size=16, hidden_size=32)
h_new, c_new = cell(x, delta_t, h, c)

print("新的隐藏状态:", h_new.shape)  # 输出: torch.Size([1, 32])
print("新的记忆单元:", c_new.shape)  # 输出: torch.Size([1, 32])
