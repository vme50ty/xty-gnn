'''
Author: lee12345 15116908166@163.com
Date: 2024-10-28 10:22:17
LastEditors: vme50ty 15116908166@163.com
LastEditTime: 2025-03-13 13:37:56
FilePath: /Gnn/DHGNN-LSTM/Codes/src/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch

class Config():
    def __init__(self):
        self.dataPath="../normal"
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_weights=[1.0, 1.0,1.0,1.0] ## 正常用户、直接攻击者、慢速攻击者、引导DDoS的攻击者
        self.graph_nums=10
        self.input_dim = 128
        self.hidden_dim = 256
        self.learning_rate = 0.001
        self.epochs=100
        self.test_ratio=0.2
        self.batch_size=1
        