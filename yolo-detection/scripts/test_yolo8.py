#!/usr/bin/env python3
from ultralytics import YOLO

def main():
    model = YOLO('/root/shared-nvme/endoscope/runs/yolo8_finetune/weights/best.pt')
    
    results = model.val(
        data='/root/shared-nvme/split-yolo-data/data.yaml',
        split='test',
        device='0',
    )
    
    print("\n===== Test Results =====")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"Precision: {results.box.mp:.4f}")
    print(f"Recall: {results.box.mr:.4f}")

if __name__ == '__main__':
    main()
