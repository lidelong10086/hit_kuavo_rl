import os
import shutil
import random

# 原始图片文件夹
image_src = "YOLO_Data/handler" 
# 原始标签文件夹
label_src = "YOLO_Data/handler_txt" 
# 训练用的数据集根目录
target_root = "dataset" 

# 创建YOLO标准结构
sub_folders = ['images/train', 'images/val', 'labels/train', 'labels/val']
for folder in sub_folders:
    os.makedirs(os.path.join(target_root, folder), exist_ok=True)

# 获取所有图片文件配对
# 获取所有.jpg文件名
image_files = [f[:-4] for f in os.listdir(image_src) if f.endswith('.jpg')]
random.seed(42) # 固定随机种子
random.shuffle(image_files)

# 90%训练，10%验证
split_idx = int(len(image_files) * 0.9)
train_list = image_files[:split_idx]
val_list = image_files[split_idx:]

def process_data(file_list, subset):
    count = 0
    for name in file_list:
        img_name = name + ".jpg"
        txt_name = name + ".txt"
        
        # 检查图片和标签是否成对
        src_img_path = os.path.join(image_src, img_name)
        src_txt_path = os.path.join(label_src, txt_name)
        
        if os.path.exists(src_img_path) and os.path.exists(src_txt_path):
            # 拷贝到images/train或images/val
            shutil.copy(src_img_path, os.path.join(target_root, 'images', subset, img_name))
            # 拷贝到labels/train或labels/val
            shutil.copy(src_txt_path, os.path.join(target_root, 'labels', subset, txt_name))
            count += 1
    print(f"完成 {subset} 集: 移动了 {count} 组数据")

# 执行移动
process_data(train_list, 'train')
process_data(val_list, 'val')

print(f"\n数据集划分完成,生成了{target_root} 文件夹")