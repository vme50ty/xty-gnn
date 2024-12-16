'''
Author: lee12345 15116908166@163.com
Date: 2024-10-28 10:22:17
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-16 16:09:34
FilePath: /Gnn/DHGNN-LSTM/Codes/src/config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch

class Config():
    def __init__(self):
        self.dataPath="../../datas"
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device=torch.device("cpu")
        self.graph_nums=10