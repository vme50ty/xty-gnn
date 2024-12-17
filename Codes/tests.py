'''
Author: lee12345 15116908166@163.com
Date: 2024-11-19 10:20:47
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-12-17 15:38:34
FilePath: /Gnn/DHGNN-LSTM/Codes/tests.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from src import RedisConn

redis_conn = RedisConn()
redis_conn.monitor_redis("../datas")

# proxies,proxy_user = redis_conn.get_all_proxies()
# proxy_ids=redis_conn.save_proxy_to_csv(proxies, filename="proxies.csv")
# users=redis_conn.get_all_users()
# redis_conn.save_user_to_csv(users,proxy_user,proxy_ids)
