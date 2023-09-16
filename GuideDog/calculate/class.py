import pandas as pd
import numpy as np
import glob  # 用于匹配多个文件

# 定义分类函数
def define_classed(m1, m2):
    if np.logical_or(abs(m1) < 200, abs(m2) < 200):
        return 0  # 停止
    elif m1 >= 200 and m2 >= 200 and abs(m1 - m2) < 200:
        return 1  # 前进
    elif m1 <= -200 and m2 <= -200 and abs(m1 - m2) < 200:
        return 2  # 后退
    elif m1 - m2 >= 200:
        return 3  # 右转
    else:
        return 4  # 左转

# 统计不同类别的数目
stop_count = 0
forward_count = 0
backward_count = 0
right_turn_count = 0
left_turn_count = 0

# 匹配多个CSV文件
csv_files = glob.glob('*.csv')  # 匹配当前目录下的所有CSV文件

for file in csv_files:
    # 读取CSV文件
    df = pd.read_csv(file)

    for index, row in df.iterrows():
        m1 = row['m1']
        m2 = row['m2']
        class_label = define_classed(m1, m2)
        if class_label == 0:
            stop_count += 1
        elif class_label == 1:
            forward_count += 1
        elif class_label == 2:
            backward_count += 1
        elif class_label == 3:
            right_turn_count += 1
        elif class_label == 4:
            left_turn_count += 1

# 打印统计结果
print("停止:", stop_count)
print("前进:", forward_count)
print("后退:", backward_count)
print("右转:", right_turn_count)
print("左转:", left_turn_count)
