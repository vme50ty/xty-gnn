'''
Author: lee12345 15116908166@163.com
Date: 2024-12-23 15:08:21
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-25 16:03:55
FilePath: /Gnn/DHGNN-LSTM/Codes/src/makeData.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import random
import csv
import os
# 用户类
class User:
    def __init__(self, user_id, belong_proxy=1, is_attacker=False):
        self.id = user_id
        self.userIP = f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        self.requestCount = 0
        self.avg_latency = 0
        self.requestPerMinute = 0
        self.connsCount = random.randint(0, 5)
        self.credit = 0
        self.belong = belong_proxy
        self.is_attacker = is_attacker

    def generate_behavior(self):
        """普通用户的行为生成"""
        self.requestCount = random.randint(0, 50)
        self.avg_latency = random.uniform(0.1, 1)
        self.requestPerMinute = random.randint(1, 10)
        self.credit = random.randint(1, 5)
        
        
        
class DirectAttacker(User):
    def generate_behavior(self):
        """直接攻击者：高请求数量，高信用"""
        self.requestCount = random.randint(1000, 2000)
        self.avg_latency = random.uniform(0.5, 2)
        self.requestPerMinute = random.randint(50, 200)
        self.credit = random.randint(20, 30)
                            

class SlowAttacker(User):
    def generate_behavior(self):
        """慢速攻击者：正常请求数，资源消耗增加"""
        self.requestCount = random.randint(20, 50)
        self.avg_latency = random.uniform(0.5, 5)  # 高时延
        self.requestPerMinute = random.randint(1, 10)
        self.credit = random.randint(1, 10)
        self.connsCount = random.randint(10, 20)  # 资源占用提高

class StealthAttacker(User):
    def generate_behavior(self):
        """高隐蔽攻击者：与普通用户行为相同，但所在代理异常"""
        super().generate_behavior()

# 代理类
class Proxy:
    def __init__(self, proxy_id):
        self.id = proxy_id
        self.proxyname = f"proxy{proxy_id}"
        self.SEND_KILOBYTES = random.uniform(10, 15)
        self.DISK_LOAD = random.uniform(3.5, 4)
        self.CPU_LOAD = random.uniform(10, 30)
        self.RECEIVE_KILOBYTES = random.uniform(6, 8)
        self.MEM_LOAD = random.uniform(70, 75)
        self.DISK_LOAD_COUNT = random.randint(20, 50)
        self.users = []  # 代理管理的用户
        self.UserNum=0

    def add_user(self, user):
        self.users.append(user)
        self.UserNum+=1
        
    def update_load(self):
        """根据用户行为更新代理负载"""
        self.SEND_KILOBYTES = sum(user.requestCount * 0.01 for user in self.users)
        self.RECEIVE_KILOBYTES =  sum(user.requestCount * 0.001 for user in self.users)
        self.CPU_LOAD = min(100,self.CPU_LOAD+sum(user.requestPerMinute * 0.0001 for user in self.users))
        self.DISK_LOAD = min(100,self.DISK_LOAD+random.uniform(0,0.01) * len(self.users)) #最大值应该为100
        self.MEM_LOAD = min(100, sum(user.connsCount * 10 for user in self.users) + 70)
        self.DISK_LOAD_COUNT += len(self.users)
        
        # 检查是否存在特定类型的攻击者
        for user in self.users:
            user.belong_proxy=self.id
            if isinstance(user, SlowAttacker):
                # 慢速攻击者的影响
                self.SEND_KILOBYTES += 0.1 * user.requestCount
                self.DISK_LOAD_COUNT += 100  # 每个慢速攻击者增加的磁盘访问次数
                self.CPU_LOAD = min(100, self.CPU_LOAD + random.uniform(10, 50))
                self.MEM_LOAD = min(100, self.MEM_LOAD + random.uniform(10, 30))
                
            if isinstance(user, StealthAttacker):
                self.CPU_LOAD = min(100, self.CPU_LOAD + random.uniform(10, 30))
                self.MEM_LOAD = min(100, self.MEM_LOAD + random.uniform(15, 35))
                self.RECEIVE_KILOBYTES += random.uniform(10, 40)
                self.SEND_KILOBYTES += random.uniform(15, 35)
                
    def show_status(self):
        """打印代理的当前状态"""
        print(f"Proxy {self.proxyname} 状态:")
        print(f"  SEND_KILOBYTES: {self.SEND_KILOBYTES:.2f}")
        print(f"  RECEIVE_KILOBYTES: {self.RECEIVE_KILOBYTES:.2f}")
        print(f"  CPU_LOAD: {self.CPU_LOAD:.2f}")
        print(f"  DISK_LOAD: {self.DISK_LOAD:.2f}")
        print(f"  MEM_LOAD: {self.MEM_LOAD:.2f}")
        print(f"  DISK_LOAD_COUNT: {self.DISK_LOAD_COUNT}")
        print(f"  管理用户数: {self.UserNum}")
        


# 数据生成函数
def generate_users(num_users, attack_distribution):
    users = []
    for user_id in range(1, num_users + 1):
        attacker_type = random.choices(
            [None, DirectAttacker, SlowAttacker, StealthAttacker],
            weights=attack_distribution
        )[0]

        if attacker_type is None:
            user = User(user_id)
        else:
            user = attacker_type(user_id, is_attacker=True)

        user.generate_behavior()
        users.append(user)

    return users


def assign_proxies(users, proxies):
    for user in users:
        belong_proxy = random.choice(proxies)
        user.belong=belong_proxy.id
        belong_proxy.add_user(user)

    for proxy in proxies:
        proxy.update_load()

def save_to_csv(users, proxies,path):
    # 保存用户数据
    if not os.path.exists(path):
        os.makedirs(path)
    
    folder_prefix = 'graph'
    existing_folders = [folder for folder in os.listdir(path)]
    existing_count = len(existing_folders)
    new_folder_name = f"{folder_prefix}{existing_count }"
    folder_path = os.path.join(path, new_folder_name)
    
    os.makedirs(folder_path, exist_ok=True)
    
    user_file_path = os.path.join(folder_path, 'users.csv')
    with open(user_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'userIP', 'requestCount', 'avg_latency', 'requestPerMinute', 'connsCount', 'credit', 'belong'])
        for user in users:
            writer.writerow([user.id, user.userIP, user.requestCount, 
                             user.avg_latency, user.requestPerMinute, user.connsCount, user.credit, user.belong])

    # 保存代理数据
    proxy_file_path = os.path.join(folder_path, 'proxys.csv')
    with open(proxy_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'proxyname', 'SEND_KILOBYTES', 
                         'DISK_LOAD', 'CPU_LOAD', 'RECEIVE_KILOBYTES', 'MEM_LOAD', 'DISK_LOAD_COUNT', 'UserNum'])
        for proxy in proxies:
            writer.writerow([proxy.id, proxy.proxyname, proxy.SEND_KILOBYTES, proxy.DISK_LOAD, 
                            proxy.CPU_LOAD, proxy.RECEIVE_KILOBYTES, proxy.MEM_LOAD, proxy.DISK_LOAD_COUNT, proxy.UserNum])
            
def save_labels_to_csv(users,path, filename="label.csv"):
    """
    将用户信息存储为CSV文件，每个用户根据类型标记normal, abnormal, unknown。

    :param users: List[User] 用户列表
    :param filename: str 保存的CSV文件名
    """
    flieName=os.path.join(path,filename)
    with open(flieName, mode='w', newline='') as file:
        writer = csv.writer(file)
        # 写入CSV标题
        writer.writerow(["id", "normal", "abnormal", "unknown"])

        for user in users:
            if isinstance(user, DirectAttacker) or isinstance(user, SlowAttacker):
                # 直接攻击者和慢速攻击者为 abnormal
                writer.writerow([user.userIP, 0, 1, 0])
            elif isinstance(user, StealthAttacker):
                # 高隐蔽攻击者为 unknown
                writer.writerow([user.userIP, 0, 0, 1])
            else:
                # 普通用户为 normal
                writer.writerow([user.userIP, 1, 0, 0])

# 主程序
if __name__ == "__main__":
    # 参数设置
    num_users =100
    num_proxies = 3
    attack_distribution = [0.95, 0.05, 0.00, 0.00]  # 普通用户, 直接攻击者, 慢速攻击者, 高隐蔽攻击者的比例
    
     # 初始化代理路径
    base_path = '../../datas_Direct/'
    folder_prefix = 'data_folder'
    existing_folders = [folder for folder in os.listdir(base_path)]
    existing_count = len(existing_folders)
    new_folder_name = f"{folder_prefix}{existing_count + 1}"
    path = os.path.join(base_path, new_folder_name)
    
    # 初始化代理
    proxies = [Proxy(i) for i in range(1, num_proxies + 1)]
    
    # 生成用户
    users = generate_users(num_users, attack_distribution)
    
    for i in range (1,10):
        assign_proxies(users, proxies)
        save_to_csv(users, proxies,path)
    save_labels_to_csv(users,path)
        
        
    # 打印用户数据
    # print("Users:")
    # for user in users:
    #     print(vars(user))

    # # 打印代理数据
    # print("\nProxies:")
    # for proxy in proxies:
    #     print(vars(proxy))
