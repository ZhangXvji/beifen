import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
df = pd.read_csv('./motor_raw.csv')

# 过滤掉异常值
df = df[df['m1'] <= 10000]
df = df[df['m2'] <= 10000]

df = df[df['stamp'] < 1000].head(1000)
# 提取数据
x = df['stamp']
y1 = df['m1']
y2 = df['m2']


# 创建一个新的图形
plt.figure(figsize=(10, 6))

# 绘制第一条曲线
plt.plot(x, y1, label='sensor1', linestyle='-', marker='o', markersize=5, color='b')

# 绘制第二条曲线
plt.plot(x, y2, label='sensor2', linestyle='-', marker='o', markersize=5, color='g')



# 添加标签和标题
plt.xlabel('stamp')
plt.ylabel('Value')
plt.title('m1 and m2 vs. stamp')

# 添加图例
plt.legend()

# 显示图形
plt.grid(True)
plt.show()
