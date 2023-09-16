# BEV说明   
通过点云pcd和4个RGB相机的jpg图片生成BEV地图。  
  
  点云存放地址：../dataset/rosbag_raw/{rosbagname}/cloudpoints
  提取数据存放： ../dataset/raw/{rosbagname}/video/0 or 2 or 4 or 6  

## BEV_map.py  
rosbag包名称在代码中是需要修改的，bag_time = "修改包名称，不带.bag"  
  
  你可能需要安装yolov8的环境，请确保你的环境符合  
  Python>=3.8 environment with PyTorch>=1.8

然后即可一键安装：pip install ultralytics

