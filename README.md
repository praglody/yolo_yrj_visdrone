# Yolo v5 小目标识别

## 训练

```shell
python train.py --weights yolov5m.pt --cfg models/yolov5m_visdrone.yaml --data data/VisDrone.yaml --epochs 1 --batch-size 64 --img-size 640 --device 0
```
