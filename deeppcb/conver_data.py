import os
import random

import cv2
import time
from shutil import copyfile
from xml.dom import minidom

name_dict = {'0': 'background', '1': 'open', '2': 'short',
             '3': 'mousebite', '4': 'spur', '5': 'copper', '6': 'pin-hole'}


# def transfer_to_yolo(pic, txt, file_name, xml_save_path):

def convert(size, box):  # size:(原图w,原图h) , box:(xmin,xmax,ymin,ymax)
    dw = 1. / size[0]  # 1/w
    dh = 1. / size[1]  # 1/h
    x = (box[0] + box[1]) / 2.0  # 物体在图中的中心点x坐标
    y = (box[2] + box[3]) / 2.0  # 物体在图中的中心点y坐标
    w = box[1] - box[0]  # 物体实际像素宽度
    h = box[3] - box[2]  # 物体实际像素高度
    x = x * dw  # 物体中心点x的坐标比(相当于 x/原图w)
    w = w * dw  # 物体宽度的宽度比(相当于 w/原图w)
    y = y * dh  # 物体中心点y的坐标比(相当于 y/原图h)
    h = h * dh  # 物体宽度的宽度比(相当于 h/原图h)
    return [x, y, w, h]  # 返回 相对于原图的物体中心点的x坐标比,y坐标比,宽度比,高度比,取值范围[0-1]


if __name__ == '__main__':
    t = time.time()
    txt_folder = '../../datasets/DeepPCB'
    img_folder = '../../datasets/DeepPCB'
    train_images_folder = '../../datasets/DeepPCB/train/images'
    train_labels_folder = '../../datasets/DeepPCB/train/labels'
    val_images_folder = '../../datasets/DeepPCB/val/images'
    val_labels_folder = '../../datasets/DeepPCB/val/labels'

    if not os.path.exists(img_folder + "/train/images"):
        os.makedirs(img_folder + "/train/images")
    if not os.path.exists(img_folder + "/train/labels"):
        os.makedirs(img_folder + "/train/labels")
    if not os.path.exists(img_folder + "/val/images"):
        os.makedirs(img_folder + "/val/images")
    if not os.path.exists(img_folder + "/val/labels"):
        os.makedirs(img_folder + "/val/labels")

    with open(txt_folder + "/trainval.txt", "r") as f:
        txt_file = f.readlines()


    def gen(data_list, is_train=True):
        size = []
        for txt in data_list:
            txt_list = txt.strip().split(" ")
            print(txt_list)
            txt_full_path = os.path.join(txt_folder, txt_list[1])
            img_full_path = os.path.join(img_folder, txt_list[0].split(".")[0] + "_test.jpg")

            print(img_full_path)
            img = cv2.imread(img_full_path)
            h, w = img.shape[0], img.shape[1]

            labels = []
            with open(txt_full_path, "r") as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    p = [int(x) for x in line.strip().split()]
                    size.append((p[2]-p[0])*(p[3]-p[1]))
                    r = convert((w, h), (p[0], p[2], p[1], p[3]))
                    r.insert(0, p[4] - 1)
                    labels.append(" ".join([str(x) for x in r]))

            if is_train:
                label_file_path = os.path.join(train_labels_folder, os.path.basename(txt_full_path))
                img_file_path = os.path.join(train_images_folder, os.path.basename(txt_list[0]))
            else:
                label_file_path = os.path.join(val_labels_folder, os.path.basename(txt_full_path))
                img_file_path = os.path.join(val_images_folder, os.path.basename(txt_list[0]))
            print(label_file_path)
            print(img_file_path)
            # copyfile(img_full_path, img_file_path)
            # with open(label_file_path, "w") as f:
            #     f.write("\n".join(labels))
        size.sort()
        print(size)

    train_list = random.sample(txt_file, 800)
    val_list = [x for x in txt_file if x not in train_list]
    #gen(train_list)
    gen(val_list, False)
