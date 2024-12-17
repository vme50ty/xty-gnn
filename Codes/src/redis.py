'''
Author: lee12345 15116908166@163.com
Date: 2024-12-17 09:36:01
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-17 15:49:29
FilePath: /Gnn/DHGNN-LSTM/Codes/src/redis.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import redis
from src import Config
import csv
import json
import os

class RedisConn():
    def __init__(self):
        #连接redis
        self.config=Config()
        
        self.redisUser = redis.StrictRedis(host=self.config.redis_host, port=self.config.redis_port, db=4)
        self.redisProxy = redis.StrictRedis(host=self.config.redis_host, port=self.config.redis_port, db=5)
    
    def get_all_proxies(self)->tuple[dict, dict]:
        """
        读取 redisProxy 数据库中所有 proxy 相关内容。
        键名为 proxy 的 proxyName，每个 proxyName 包含多个键值对。
        返回一个字典，结构为：{proxyName: {key: value, ...}}
        返回proxy_user: {proxyName: [userIP1, userIP2, ...]}
        """
        result={}
        proxy_user={}
        all_keys = self.redisProxy.keys('*')
        
        for key in all_keys:
            key_str = key.decode('utf-8')  # Redis 返回的键是 bytes，需要转换为字符串

            proxy_info = self.redisProxy.hgetall(key_str)
            
            # 将proxy对应的userIP，存储到proxy_user中。即{proxy1:"163.137.94.123","93.123.192.159","19.93.159.84",proxy2:...}的形式
            # 将每个 proxy 的键值对从 bytes 转换为字符串
            proxy_info_decoded = {k.decode('utf-8'): v.decode('utf-8') for k, v in proxy_info.items()}
            
             # 保存结果
            users_str = proxy_info_decoded.pop("USERS", "[]")  # 移除 USERS 字段，同时获取其值
            
            user_ips = json.loads(users_str)
            proxy_user[key_str] = user_ips
            result[key_str] = proxy_info_decoded
            
        # print(result)
        # print("代理对应用户:", proxy_user)
        return result,proxy_user
    
    def get_all_users(self):
        result={}
        all_keys=self.redisUser.keys('*')
        
        for key in all_keys:
            key_str = key.decode('utf-8')
            user_info = self.redisUser.hgetall(key_str)
            user_info_decoded = {k.decode('utf-8'): v.decode('utf-8') for k, v in user_info.items()}
            result[key_str] = user_info_decoded
        return result
    
    
    def save_proxy_to_csv(self,data,filename="proxys.csv")-> dict:
        """
        将 proxy 数据保存为 CSV 文件。
        返回proxy_ids: {proxyName:id1,id2}
        """
        try:
        # 需要存储的所有字段名（包括 proxyname）
            headers = ["id", "proxyname", "SEND_KILOBYTES", "DISK_LOAD", "CPU_LOAD", 
                       "RECEIVE_KILOBYTES", "MEM_LOAD", "DISK_LOAD_COUNT"]
            
            # 写入 CSV 文件
            with open(filename, mode="w", newline='') as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                writer.writeheader()  # 写入表头
                
                proxyName_ids={}
                # 数据处理
                for idx, (proxy_name, proxy_data) in enumerate(data.items(), start=1):
                    row = {
                        "id": idx,
                        "proxyname": proxy_name,
                        "SEND_KILOBYTES": proxy_data.get("SEND_KILOBYTES", "0"),
                        "DISK_LOAD": proxy_data.get("DISK_LOAD", "0"),
                        "CPU_LOAD": proxy_data.get("CPU_LOAD", "0"),
                        "RECEIVE_KILOBYTES": proxy_data.get("RECEIVE_KILOBYTES", "0"),
                        "MEM_LOAD": proxy_data.get("MEM_LOAD", "0"),
                        "DISK_LOAD_COUNT": proxy_data.get("DISK_LOAD_COUNT", "0"),
                    }
                    proxyName_ids[proxy_name]=idx
                    writer.writerow(row)
            print(f"数据已成功保存到 {filename}")
        except Exception as e:
            print(f"保存到 CSV 失败: {e}")
        return proxyName_ids

    def save_user_to_csv(self, data, proxy_user, proxy_ids, filename="users.csv") -> None:
        """
        将 user 数据保存为 CSV 文件，增加 belong 字段标识用户所属的代理 id。
        :param data: 用户数据 {userIP: {key1: value1, ...}}
        :param proxy_user: 每个 proxyName 对应的 userIP 列表 {proxyName: [userIP1, userIP2, ...]}
        :param proxy_ids: proxyName 对应的代理 ID {proxyName: id}
        :param filename: 输出的 CSV 文件名
        """
        try:
            # 需要存储的所有字段名，增加 'belong' 字段
            headers = ["id", "userIP", "requestCount", "avg_latency", "requestPerMinute", 
                    "connsCount", "credit", "belong"]

            # 构建 userIP 到代理 id 的映射
            user_to_proxy_id = {}
            for proxy_name, user_ips in proxy_user.items():
                proxy_id = proxy_ids.get(proxy_name)  # 获取对应的代理 id
                if proxy_id and user_ips:
                    for user_ip in user_ips:
                        user_to_proxy_id[user_ip] = proxy_id

            # 写入 CSV 文件
            with open(filename, mode="w", newline='') as file:
                writer = csv.DictWriter(file, fieldnames=headers)
                writer.writeheader()  # 写入表头

                # 数据处理
                for idx, (user_IP, user_data) in enumerate(data.items(), start=1):
                    row = {
                        "id": idx,
                        "userIP": user_IP,
                        "requestCount": user_data.get("requestCount", "0"),
                        "avg_latency": user_data.get("avg_latency", "0"),
                        "requestPerMinute": user_data.get("requestPerMinute", "0"),
                        "connsCount": user_data.get("connsCount", "0"),
                        "credit": user_data.get("credit", "0"),
                        "belong": user_to_proxy_id.get(user_IP, "N/A")  # 获取所属代理 ID，默认 "N/A"
                    }
                    writer.writerow(row)

            print(f"用户数据已成功保存到 {filename}")
        except Exception as e:
            print(f"保存用户数据到 CSV 失败: {e}")
    
    
    def monitor_redis(self,path):
        """
        监听 Redis 的 Notify_shuffle 频道，并打印接收到的洗牌时间。
        """
        try:
            pubsub = self.redisProxy.pubsub()
            pubsub.subscribe("Notify_shuffle")  # 订阅频道

            print("Listening to Redis channel: Notify_shuffle")

            for message in pubsub.listen():
                # 过滤消息类型，只处理 'message'
                if message['type'] == 'message':
                    shuffle_time = message['data'].decode('utf-8')
                    print(f"Received shuffle time: {shuffle_time}")
                    
                    # 在path下添加名称为shuffle_time的文件夹x。
                    x = os.path.join(path, shuffle_time)
                    os.makedirs(x, exist_ok=True)
                    
                    proxies,proxy_user = self.get_all_proxies()
                    proxy_ids=self.save_proxy_to_csv(proxies, filename=os.path.join(x, "proxies.csv"))
                    users=self.get_all_users()
                    self.save_user_to_csv(users,proxy_user,proxy_ids,filename=os.path.join(x, "users.csv"))

                    
        except Exception as e:
            print(f"Error while monitoring Redis: {e}")
        