#!/usr/bin/env python3
from ultralytics import YOLO

def main():
    model = YOLO('/root/shared-nvme/endoscope/runs/yolo8_finetune/weights/best.pt')
    
    image_path = '/root/shared-nvme/split-yolo-data/images/test/0030fbc2-frame_0741.png'
    
    results = model.predict(
        source=image_path,
        device='0',
        save=True,
        save_txt=True,
        conf=0.25,
        iou=0.7,
    )
    
    print(f"图片路径: {image_path}")
    print(f"检测到 {len(results[0].boxes)} 个目标")
    
    for i, box in enumerate(results[0].boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  - 类别: {results[0].names[cls]}, 置信度: {conf:.2f}")

if __name__ == '__main__':
    main()
