import csv
import random

# 假设你的CSV文件名为 data.csv
input_csv_file = 'output.csv'
output_csv_file = 'new_data.csv'

# 打开原始CSV文件和新CSV文件
with open(input_csv_file, mode='r') as input_file, open(output_csv_file, mode='w', newline='') as output_file:
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)
    
    # 读取原始CSV文件的列名并添加到新文件
    header = next(reader)
    writer.writerow(header)
    
    # 初始化计数器和一个字典来存储每个类别的行数
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    total_rows = 0
    
    # 遍历原始CSV文件的每一行，随机删去一半class为1的数据
    for row in reader:
        class_value = int(row[-1])  # 假设类别在每行的最后一列
        
        # 如果类别不为1，则将该行写入新文件两次
        if class_value != 1:
            writer.writerow(row)
            writer.writerow(row)
            class_counts[class_value] += 2
        else:
            # 如果类别为1，则根据一定概率将其写入新文件
            if random.random() < 0.5:
                writer.writerow(row)
                class_counts[class_value] += 1
        
        total_rows += 1

# 打印每个类别的行数和总行数
for class_value, count in class_counts.items():
    print(f"Class {class_value}: {count} rows")
print(f"Total rows: {total_rows}")

print("New CSV file has been created:", output_csv_file)
