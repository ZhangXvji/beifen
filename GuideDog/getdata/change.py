import pandas as pd
import os
import numpy as np
def filter_motor(bag_time):
    # 读取CSV文件
    df = pd.read_csv(f'../dataset/raw/{bag_time}/ecparm/motor/motor_raw.csv')

    # # 处理m1列
    # m1_values = df['m1'].values
    # m1_abs = abs(m1_values)
    # m1_mask = m1_abs > 1000

    # for i in range(1, len(m1_values) - 1):
    #     if m1_mask[i]:
    #         m1_values[i] = (m1_values[i - 1] + m1_values[i + 1]) / 2

    # df['m1'] = m1_values

    # # 处理m2列
    # m2_values = df['m2'].values
    # m2_abs = abs(m2_values)
    # m2_mask = m2_abs > 1000

    # for i in range(1, len(m2_values) - 1):
    #     if m2_mask[i]:
    #         m2_values[i] = (m2_values[i - 1] + m2_values[i + 1]) / 2

    # df['m2'] = m2_values

    # 处理m1列
    m1_values = df['m1'].values

    for i in range(2, len(m1_values) - 2):
        avg = (m1_values[i - 1] + m1_values[i + 1]) / 2
        
        if abs(m1_values[i]-m1_values[i-1]) > 500 :
            m1_values[i] = avg

    df['m1'] = m1_values

    # 处理m2列
    m2_values = df['m2'].values

    for j in range(2, len(m2_values) - 2):
        avg = (m2_values[i - 1] + m2_values[i + 1]) / 2
        
        if abs(m2_values[i]-m2_values[i-1]) > 500 :
            m2_values[i] = avg

    df['m2'] = m2_values

    # 保存修改后的数据到新的CSV文件
    df.to_csv(f'../dataset/raw/{bag_time}/ecparm/motor/motor_raw.csv', index=False)

if __name__ == "__main__":
    for file_name in os.listdir("../dataset/raw"):
        name_without_extension = os.path.splitext(file_name)[0]
        bag_time = name_without_extension
        filter_motor(bag_time)