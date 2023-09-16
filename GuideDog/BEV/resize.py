from PIL import Image

# 打开图片
img = Image.open("/home/guide/GuideDog/dataset/raw/2023-08-19-20-14-45/video/6/2.jpg")  # 替换为您的图片文件名

# 调整图片大小为 56x56
new_size = (56, 56)
resized_img = img.resize(new_size)

# 保存调整后的图片
resized_img.save("/home/guide/GuideDog/dataset/raw/2023-08-19-20-14-45/video/6/2.jpg")  # 保存为新的文件名，可以替换为您喜欢的名字
