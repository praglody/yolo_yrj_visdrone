# Yolo v5 小目标识别

## 训练

```shell
python train.py --weights yolov5s.pt --cfg models/yolov5s_visdrone.yaml --data data/VisDrone.yaml --hyp data/hyps/hyp_yrj_visdrone.yaml --epochs 60 --batch-size 32 --img-size 640 --device 0
```
