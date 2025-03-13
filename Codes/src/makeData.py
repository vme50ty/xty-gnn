'''
Author: lee12345 15116908166@163.com
Date: 2024-12-23 15:08:21
LastEditors: vme50ty 15116908166@163.com
LastEditTime: 2025-03-13 00:31:49
FilePath: /Gnn/DHGNN-LSTM/Codes/src/makeData.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import random
import csv
import os,time
import numpy as np

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
        self.credit = random.randint(1, 8)
        
        
        
class DirectAttacker(User):
    def generate_behavior(self):
        """直接攻击者：高请求数量，高信用"""
        self.requestCount = random.randint(1000, 2000)
        self.avg_latency = random.uniform(0.5, 2)
        self.requestPerMinute = random.randint(50, 200)
        self.credit = random.randint(5, 10)
                            

class SlowAttacker(User):
    def generate_behavior(self):
        """慢速攻击者：随机提高两个指标"""
        super().generate_behavior()  # 先生成普通用户的行为
        
        # 可被提升的指标
        possible_features = ["requestCount", "avg_latency", "requestPerMinute", "connsCount"]
        chosen_features = random.sample(possible_features, random.randint(2,4))  # 随机选择两个指标提高
        
        if "requestCount" in chosen_features:
            self.requestCount = random.randint(30,50)  # 提高请求数

        if "avg_latency" in chosen_features:
            self.avg_latency = random.uniform(0.5, 2)  # 提高时延

        if "requestPerMinute" in chosen_features:
            self.requestPerMinute = random.randint(5, 15)  # 提高请求速率
        
        if "connsCount" in chosen_features:
            self.connsCount = random.randint(5, 8)  # 增加并发连接

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
        self.CPU_LOAD = random.uniform(10, 20)
        self.RECEIVE_KILOBYTES = random.uniform(6, 8)
        self.MEM_LOAD = random.uniform(70, 75)
        self.DISK_LOAD_COUNT = random.randint(20, 50)
        self.users = []  # 代理管理的用户
        self.UserNum=0
        self.beingAttack=0

    def add_user(self, user):
        self.users.append(user)
        self.UserNum+=1
        
    def reset_load(self):
        """重置代理的负载数据"""
        self.SEND_KILOBYTES = 0
        self.RECEIVE_KILOBYTES = 0
        self.CPU_LOAD = 0
        self.DISK_LOAD = 0
        self.MEM_LOAD = 0
        self.DISK_LOAD_COUNT = 0
        self.beingAttack=0
        self.users=[]
        
    def update_load(self):
        """根据用户行为更新代理负载"""
        self.SEND_KILOBYTES = sum(user.requestCount * 0.01 for user in self.users)
        self.RECEIVE_KILOBYTES =  sum(user.requestCount * 0.001 for user in self.users)
        self.CPU_LOAD = min(100,random.uniform(10, 20)+sum(user.requestPerMinute * 0.0001 for user in self.users))
        self.DISK_LOAD = min(100,random.uniform(10, 20)+random.uniform(0,0.01) * len(self.users)) #最大值应该为100
        self.MEM_LOAD = min(100, sum(user.connsCount * 0.01 for user in self.users) + 10)
        self.DISK_LOAD_COUNT += len(self.users)
        
        # 检查是否存在特定类型的攻击者
        for user in self.users:
            user.belong_proxy=self.id
            if isinstance(user, SlowAttacker):
                # 慢速攻击者的影响
                # self.beingAttack=1
                self.SEND_KILOBYTES += 1 * user.requestCount
                self.DISK_LOAD_COUNT += 20 # 每个慢速攻击者增加的磁盘访问次数
                self.CPU_LOAD = self.CPU_LOAD + random.uniform(50, 100)
                self.MEM_LOAD = self.MEM_LOAD + random.uniform(50, 100)
                
            if isinstance(user, StealthAttacker):
                self.CPU_LOAD = self.CPU_LOAD + random.uniform(30, 50)
                self.MEM_LOAD = self.MEM_LOAD + random.uniform(30,50)
                self.RECEIVE_KILOBYTES += random.uniform(50, 80)
                self.SEND_KILOBYTES += random.uniform(50, 80)
                self.beingAttack=1
            
            if isinstance(user,DirectAttacker):
                self.beingAttack=1

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
    for proxy in proxies:
        proxy.UserNum=0
        proxy.reset_load()  # 重置代理的负载
        
    for user in users:
        belong_proxy = random.choice(proxies)
        user.belong=belong_proxy.id
        belong_proxy.add_user(user)

    for proxy in proxies:
        proxy.update_load()

def save_to_csv(users, proxies, path):
    # 创建存储目录
    if not os.path.exists(path):
        os.makedirs(path)
    
    current_timestamp = int(time.time())
    existing_folders = [folder for folder in os.listdir(path)]
    if not existing_folders:
        new_folder_name = str(current_timestamp)
    else:
        # 找到最新的文件夹（按时间戳排序）
        existing_folders = sorted(existing_folders, key=int)
        latest_folder = existing_folders[-1]
        attacked_proxies = [p for p in proxies if p.beingAttack == 1]
        if attacked_proxies:
            # 若有攻击代理，在最新时间戳上加 3~8 秒
            new_timestamp = int(latest_folder) + random.randint(3, 8)
        else:
            # 若无攻击代理，在最新时间戳上加 15~25 秒
            new_timestamp = int(latest_folder) + random.randint(15, 25)
        new_folder_name = str(new_timestamp)
        
    # 创建新文件夹
    folder_path = os.path.join(path, new_folder_name)
    os.makedirs(folder_path, exist_ok=True)
    

    # 计算健康代理的中位数基准值
    healthy_proxies = [p for p in proxies if p.beingAttack == 0]
    
    if len(healthy_proxies)==0:
        print("全部代理遭受攻击，坏")
    # 定义中位数计算函数（处理空值情况）
    
    def safe_median(values, default=1):
        if len(values) == 0:
            print("不存在值")
            return default  # 避免空列表问题
        return np.median(values).item()  # 确保返回 Python 数值
    
    # 计算各指标中位数
    medians = {
        'SEND': safe_median([p.SEND_KILOBYTES for p in healthy_proxies]),
        'DISK': safe_median([p.DISK_LOAD for p in healthy_proxies]),
        'CPU': safe_median([p.CPU_LOAD for p in healthy_proxies]),
        'RECEIVE': safe_median([p.RECEIVE_KILOBYTES for p in healthy_proxies]),
        'MEM': safe_median([p.MEM_LOAD for p in healthy_proxies]),
        'DISK_COUNT': safe_median([p.DISK_LOAD_COUNT for p in healthy_proxies])
    }

    # 保存用户数据（保持不变）
    user_file_path = os.path.join(folder_path, 'users.csv')
    with open(user_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','userIP','requestCount','avg_latency','requestPerMinute','connsCount','credit','belong'])
        for u in users:
            writer.writerow([u.id, u.userIP, u.requestCount, u.avg_latency, 
                            u.requestPerMinute, u.connsCount, u.credit, u.belong])

    # 保存代理数据（存储比值）
    proxy_file_path = os.path.join(folder_path, 'proxys.csv')
    with open(proxy_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id','proxyname','SEND_ratio','DISK_ratio',
                        'CPU_ratio','RECEIVE_ratio','MEM_ratio',
                        'DISK_COUNT_ratio','UserNum','BeingAttack'])
        
        for p in proxies:
            # 计算各指标比值（保留2位小数）
            row = [
                p.id,
                p.proxyname,
                round(p.SEND_KILOBYTES / medians['SEND'], 2),
                round(p.DISK_LOAD / medians['DISK'], 2),
                round(p.CPU_LOAD / medians['CPU'], 2),
                round(p.RECEIVE_KILOBYTES / medians['RECEIVE'], 2),
                round(p.MEM_LOAD / medians['MEM'], 2),
                round(p.DISK_LOAD_COUNT / medians['DISK_COUNT'], 2),
                p.UserNum,
                p.beingAttack
            ]
            writer.writerow(row)
            
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
        writer.writerow(["id", "normal", "direct", "slow","steal"])

        for user in users:
            if isinstance(user, DirectAttacker):
                # 直接攻击者为direct
                writer.writerow([user.userIP, 0, 1, 0, 0])
            elif isinstance(user, StealthAttacker):
                # 高隐蔽攻击者为 steal
                writer.writerow([user.userIP, 0, 0, 0, 1])
                # 慢速攻击者为 slowries
            elif  isinstance(user, SlowAttacker):
                writer.writerow([user.userIP, 0, 0, 1, 0])
            else:
                # 普通用户为 normal
                writer.writerow([user.userIP, 1, 0, 0, 0])

# 主程序
if __name__ == "__main__":
    # 参数设置
    num_users =500
    num_proxies = 35
    attack_distribution = [0.97, 0.00, 0.01, 0.02]  # 普通用户, 直接攻击者, 慢速攻击者, 高隐蔽攻击者的比例
    
     # 初始化代理路径
    base_path = '../../datas_combine/'
    folder_prefix = 'data_folder'
    existing_folders = [folder for folder in os.listdir(base_path)]
    existing_count = len(existing_folders)
    new_folder_name = f"{folder_prefix}{existing_count + 1}"
    path = os.path.join(base_path, new_folder_name)
    
    # 初始化代理
    proxies = [Proxy(i) for i in range(1, num_proxies + 1)]
    
    # 生成用户
    users = generate_users(num_users, attack_distribution)
    
    for i in range (1,random.randint(9,15)):
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
