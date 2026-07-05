#!/usr/bin/env python3
import os
from ultralytics import YOLO

def main():
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='/root/shared-nvme/split-yolo-data/data.yaml',
        epochs=100,
        imgsz=640,
        batch=32,
        device='0',
        project='/root/shared-nvme/endoscope/runs',
        name='yolo8_finetune',
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
        val=True,
        workers=4,
        amp=False,
    )
    
    print("Training completed!")
    print(f"Results saved to: {results.save_dir}")

if __name__ == '__main__':
    main()
