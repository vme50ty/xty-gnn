'''
Author: lee12345 15116908166@163.com
Date: 2024-11-20 09:26:58
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-11-20 09:40:17
FilePath: /Gnn/DHGNN-LSTM/Codes/train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AEf
'''
from src import Config
from src import LoadHeteroGraph
from src import CombinedModel

def train_combined_model(model:CombinedModel,data_loader:LoadHeteroGraph,optimizer,criterion,device,epochs=10):
    model.to(device)
    model.train()
    
    epoch_losses=[]
    
    for epoch in range (epochs):
        print(f'Epcoh{epoch+1}/{epochs}')
        epoch_loss=0.0

        user_time_sequences=[]
        time_deltas=[]
        last_time=None