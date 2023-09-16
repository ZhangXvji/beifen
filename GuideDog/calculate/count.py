import pandas as pd
from collections import Counter
import csv
import torch
import ast



csv_file = "new_data.csv"
df = pd.read_csv(csv_file)

label_dict = {}


# 获取数据的行数（总数据数量）
data_count = df.shape[0]
# 打印总数据数量
print("Total data count:", data_count)


# 打开CSV文件并加载数据
with open(csv_file, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        label_str = row['label']
        # 去掉字符串中的 "tensor" 和多余的空格，并使用 ast.literal_eval 转换为 Python 对象
        label_str = label_str.replace('tensor', '').replace(' ', '').replace('\n', '')
        label_list = ast.literal_eval(label_str)
        # 将列表转换为 PyTorch Tensor
        label_tensor = torch.tensor(label_list)
        label_tensor = torch.argmax(label_tensor, dim=1)
        # 使用 Counter 统计每个值的出现次数
        value_counts = Counter(label_tensor)
        # 找到出现次数最多的值，如果有多个相同的值，取较大的
        most_common_value = max(value_counts, key=lambda x: (value_counts[x], x))


        # 将数据添加到字典中
        if str(most_common_value) in label_dict.keys():
            label_dict[str(most_common_value)].append(label_tensor)
        else:
            label_dict[str(most_common_value)] = [label_tensor]

# 遍历字典的键，并打印每个键对应值的数量
for key in label_dict.keys():
    print(f"key-{key} : {len(label_dict[key])}")


