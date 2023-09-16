# getdata说明   
本文件夹中的程序用于从录制的rosbag中提取原始数据  
注意：认为rosbag的存放地址和提取原始的存放地址都是固定的  
  rosbag存放：../dataset/rosbag_raw/  
  提取数据存放： ../dataset/raw/{rosbagname}/  
## getdata_raw.py
用于提取单个rosbag，注意在代码中手动修改rosbag名称。
## test.py  
自动提取所有rosbag。  
  
注：对缓慢的运行速度要有心里预期。
