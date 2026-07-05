import os
import shutil
import argparse

def merge_datasets(dataset1, dataset2, output):
    # 创建输出目录结构
    os.makedirs(os.path.join(output, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output, 'labels'), exist_ok=True)
    
    # 复制classes.txt文件（从dataset1复制，因为两个文件内容相同）
    shutil.copy2(os.path.join(dataset1, 'classes.txt'), os.path.join(output, 'classes.txt'))
    
    # 复制dataset1的图片
    print(f"Copying images from {dataset1}...")
    for img_file in os.listdir(os.path.join(dataset1, 'images')):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(dataset1, 'images', img_file)
            dst = os.path.join(output, 'images', img_file)
            shutil.copy2(src, dst)
            
            # 检查是否有对应的label文件
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_src = os.path.join(dataset1, 'labels', label_file) if os.path.exists(os.path.join(dataset1, 'labels')) else None
            if label_src and os.path.exists(label_src):
                label_dst = os.path.join(output, 'labels', label_file)
                shutil.copy2(label_src, label_dst)
    
    # 复制dataset2的图片
    print(f"Copying images from {dataset2}...")
    for img_file in os.listdir(os.path.join(dataset2, 'images')):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            src = os.path.join(dataset2, 'images', img_file)
            dst = os.path.join(output, 'images', img_file)
            # 检查是否有重名文件
            if os.path.exists(dst):
                # 如果有重名，添加后缀
                base_name, ext = os.path.splitext(img_file)
                counter = 1
                while os.path.exists(os.path.join(output, 'images', f"{base_name}_{counter}{ext}")):
                    counter += 1
                dst = os.path.join(output, 'images', f"{base_name}_{counter}{ext}")
            shutil.copy2(src, dst)
            
            # 检查是否有对应的label文件
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_src = os.path.join(dataset2, 'labels', label_file) if os.path.exists(os.path.join(dataset2, 'labels')) else None
            if label_src and os.path.exists(label_src):
                # 如果图片文件名被修改，对应的label文件名也需要修改
                if dst != os.path.join(output, 'images', img_file):
                    label_file = os.path.splitext(os.path.basename(dst))[0] + '.txt'
                label_dst = os.path.join(output, 'labels', label_file)
                shutil.copy2(label_src, label_dst)
    
    # 生成data.yaml配置文件
    generate_data_yaml(output)
    
    print(f"Merged dataset created at {output}")
    print(f"Total images: {len(os.listdir(os.path.join(output, 'images')))}")
    print(f"Total labels: {len(os.listdir(os.path.join(output, 'labels')))}")

def generate_data_yaml(output):
    # 读取classes.txt文件获取类别信息
    classes = []
    with open(os.path.join(output, 'classes.txt'), 'r') as f:
        for line in f:
            if line.strip():
                # 提取类别名称（去掉编号和箭头）
                class_name = line.split('→')[-1].strip()
                classes.append(class_name)
    
    # 生成data.yaml文件
    yaml_content = f"""# YOLO Dataset Configuration

# Paths
path: {os.path.abspath(output)}  # dataset root dir

# Train/val/test splits
# train: images/train
# val: images/val
# test: images/test

# Classes
nc: {len(classes)}  # number of classes
names: {classes}  # class names
"""
    
    with open(os.path.join(output, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print(f"Generated data.yaml configuration file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two YOLO datasets")
    parser.add_argument("dataset1", help="Path to first dataset")
    parser.add_argument("dataset2", help="Path to second dataset")
    parser.add_argument("output", help="Path to output merged dataset")
    args = parser.parse_args()
    
    merge_datasets(args.dataset1, args.dataset2, args.output)
