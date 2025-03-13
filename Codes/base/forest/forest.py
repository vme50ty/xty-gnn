import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import shuffle

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
        self.avg_latency = random.uniform(0.1, 2)
        self.requestPerMinute = random.randint(1, 15)
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
        chosen_features = random.sample(possible_features, 2)  # 随机选择两个指标提高
        
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

# 生成数据集
def generate_dataset(num_users=1500):
    data = []
    labels = []
    
    # 生成普通用户
    for i in range(int(num_users *0.97)):
        user = User(i)
        user.generate_behavior()
        data.append([user.requestCount, user.avg_latency, user.requestPerMinute, user.connsCount, user.credit])
        labels.append(0)  # 0 表示正常用户
    
    # 生成攻击者（每种类型占 1/6）
    for i in range(int(num_users *0.01)):
        attacker = DirectAttacker(i + num_users)
        attacker.generate_behavior()
        data.append([attacker.requestCount, attacker.avg_latency, attacker.requestPerMinute, attacker.connsCount, attacker.credit])
        labels.append(1)  

        attacker = SlowAttacker(i + num_users * 2)
        attacker.generate_behavior()
        data.append([attacker.requestCount, attacker.avg_latency, attacker.requestPerMinute, attacker.connsCount, attacker.credit])
        labels.append(2) 

        attacker = StealthAttacker(i + num_users * 3)
        attacker.generate_behavior()
        data.append([attacker.requestCount, attacker.avg_latency, attacker.requestPerMinute, attacker.connsCount, attacker.credit])
        labels.append(3) 

    # 将数据和标签组合在一起后进行打乱
    data, labels = shuffle(data, labels, random_state=42)

    return np.array(data), np.array(labels)

# 训练分类器并评估
def train_and_evaluate():
    X, y = generate_dataset(5000)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

train_and_evaluate()
