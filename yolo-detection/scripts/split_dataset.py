import os
import shutil
import random
import argparse

def split_dataset(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 检查比例是否正确
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    # 创建输出目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
    
    # 复制classes.txt文件
    shutil.copy2(os.path.join(input_dir, 'classes.txt'), os.path.join(output_dir, 'classes.txt'))
    
    # 获取所有图片文件
    image_files = []
    for file in os.listdir(os.path.join(input_dir, 'images')):
        if file.endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(file)
    
    # 打乱文件顺序
    random.shuffle(image_files)
    
    # 计算各部分的数量
    total = len(image_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    
    # 分配文件
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count+val_count]
    test_files = image_files[train_count+val_count:]
    
    # 复制文件到对应目录
    splits = [('train', train_files), ('val', val_files), ('test', test_files)]
    
    for split_name, files in splits:
        print(f"Copying {len(files)} files to {split_name} split...")
        for file in files:
            # 复制图片文件
            img_src = os.path.join(input_dir, 'images', file)
            img_dst = os.path.join(output_dir, 'images', split_name, file)
            shutil.copy2(img_src, img_dst)
            
            # 复制对应的标签文件
            label_file = os.path.splitext(file)[0] + '.txt'
            label_src = os.path.join(input_dir, 'labels', label_file)
            if os.path.exists(label_src):
                label_dst = os.path.join(output_dir, 'labels', split_name, label_file)
                shutil.copy2(label_src, label_dst)
    
    # 生成data.yaml配置文件
    generate_data_yaml(output_dir)
    
    print(f"Dataset split completed!")
    print(f"Total files: {total}")
    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")

def generate_data_yaml(output_dir):
    # 读取classes.txt文件获取类别信息
    classes = []
    with open(os.path.join(output_dir, 'classes.txt'), 'r') as f:
        for line in f:
            if line.strip():
                # 提取类别名称（去掉编号和箭头）
                class_name = line.split('→')[-1].strip()
                classes.append(class_name)
    
    # 生成data.yaml文件
    yaml_content = f"""# YOLO Dataset Configuration

# Paths
path: {os.path.abspath(output_dir)}  # dataset root dir

# Train/val/test splits
train: images/train
val: images/val
test: images/test

# Classes
nc: {len(classes)}  # number of classes
names: {classes}  # class names
"""
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
    print(f"Generated data.yaml configuration file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument("input", help="Path to input dataset")
    parser.add_argument("output", help="Path to output split dataset")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--val", type=float, default=0.1, help="Validation ratio (default: 0.1)")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio (default: 0.1)")
    args = parser.parse_args()
    
    split_dataset(args.input, args.output, args.train, args.val, args.test)
