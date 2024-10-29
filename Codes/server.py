'''
Author: lee12345 15116908166@163.com
Date: 2024-10-28 09:48:26
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2024-10-29 10:38:17
FilePath: /Gnn/DHGNN-LSTM/Codes/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

from flask import Flask, request, jsonify
import pandas as pd
import time
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])

def upload():
    # 接收 JSON 数据
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data received"}), 400

    time_now=time.time()
    formatted_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time_now))
    folder_name = f'data_folder_{formatted_time}'
    os.makedirs(folder_name, exist_ok=True)

    # 检查并保存各表格
    for table_name, table_data in data.items():
        if isinstance(table_data, list):
            # 将表格数据保存为 CSV 文件
            df = pd.DataFrame(table_data)
            file_path = os.path.join(folder_name, f"{table_name}.csv")
            df.to_csv(file_path, index=False)

    return jsonify({"message": f"Data saved to {folder_name}"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)