# Yolo v5 小目标识别

## 训练

```shell
python train.py --weights yolov5m.pt --cfg models/yolov5m_visdrone.yaml \
        --data data/VisDrone.yaml --hyp data/hyps/hyp_yrj_visdrone.yaml  \
        --epochs 32 --batch-size 32 --img-size 640 --device 0
```
