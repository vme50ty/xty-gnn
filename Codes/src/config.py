'''
Author: lee12345 15116908166@163.com
Date: 2024-10-28 10:22:17
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-25 15:45:32
FilePath: /Gnn/DHGNN-LSTM/Codes/src/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch

class Config():
    def __init__(self):
        self.dataPath="../datas_Direct"
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.graph_nums=10
        self.input_dim = 256
        self.hidden_dim = 512
        self.learning_rate = 0.001
        self.epochs=1000
        
        self.redis_host="10.29.202.222"
        self.redis_port="6379"