import utils.autoanchor as autoAC

# 对数据集重新计算 anchors

new_anchors = autoAC.kmean_anchors(
    dataset='data/VisDrone.yaml',  # 数据集配置文件路径。
    n=9,  # anchors 组数量
    img_size=640,  # 图片尺寸
    thr=5.0,  # dataset中标注框宽高比最大阈值,在超参文件中"anchor_t"设置。
    gen=1000,  # kmean算法iter次数
    verbose=True  # 是否打印结果
)
print(new_anchors)
