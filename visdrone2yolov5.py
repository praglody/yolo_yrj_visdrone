import math
import random
import cv2
from utils.general import os, Path

cls_name = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
cls_count = {0: 79337, 1: 27059, 2: 10480, 3: 144867, 4: 24956, 5: 12875, 6: 4812, 7: 3246, 8: 5926, 9: 29647}
enhance_cls = [2, 5, 6, 7, 8]
# for i in range(0, 10):
#     print(f"{cls_name[i]}: {cls_count[i]}")
# bicycle: 		    10480		2.3812977099236643
# truck: 		    12875		1.9383300970873787
# tricycle: 		4812		5.186201163757273
# awning-tricycle: 	3246		7.688231669747381
# bus: 		        5926		4.211272359095512

for i in range(0, 10):
    if cls_count[i] < 24956:
        print(f"{cls_name[i]}: \t\t{cls_count[i]}\t\t{24956 / cls_count[i]}")


def get_enhance_num(cls):
    p = {
        2: 2.3812977099236643,
        5: 1.9383300970873787,
        6: 5.186201163757273,
        7: 7.688231669747381,
        8: 4.211272359095512,
    }
    return math.ceil(p[cls])


def val_start_position(label_list, x01, y01, x02, y02):
    for lab in label_list:
        x11, y11, x12, y12 = lab[1], lab[2], lab[1] + lab[3], lab[2] + lab[4]
        lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
        ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
        sax = abs(x01 - x02)
        sbx = abs(x11 - x12)
        say = abs(y01 - y02)
        sby = abs(y11 - y12)
        if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
            return False
    return True


def enhance_data(label_list, img_size, src_img):
    label_list_clone = label_list.copy()

    for lab in label_list_clone:
        if lab[0] not in enhance_cls:
            continue
        enhance_num = get_enhance_num(lab[0])
        for i in range(enhance_num):
            j = 1000
            while j > 0:
                j -= 1
                x1 = random.randint(0, img_size[0] - lab[3])
                y1 = random.randint(0, img_size[1] - lab[4])
                x2 = x1 + lab[3]
                y2 = y1 + lab[4]
                # 是否可以粘贴
                if val_start_position(label_list, x1, y1, x2, y2):
                    label_list.append([lab[0], x1, y1, lab[3], lab[4]])
                    sx1, sy1, sx2, sy2 = lab[1], lab[2], lab[1] + lab[3], lab[2] + lab[4]
                    src_block = src_img[sy1:sy2, sx1:sx2].copy()
                    rd = random.randint(-1, 2)  # 对图像做翻转
                    if rd != 2:
                        src_block = cv2.flip(src_block, rd)
                    src_img[y1:y2, x1:x2] = src_block
                    break
            # if j == 0:
            #     for lab in label_list:
            #         if lab[0] in enhance_cls:
            #             color = (0, 255, 0)
            #         else:
            #             color = (0, 0, 255)
            #         cv2.rectangle(src_img, (lab[1], lab[2]), (lab[1] + lab[3], lab[2] + lab[4]), color, 1)
            #     cv2.imshow("sgd", src_img)
            #     cv2.waitKey(0)
    return label_list


def visdrone2yolo(dir):
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        # size (w, h)
        # box (x, y, w, h)
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    target_folder = Path(str(dir).replace(os.sep + 'VisDrone' + os.sep, os.sep + 'VisDronePro' + os.sep))
    (target_folder / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
    (target_folder / 'images').mkdir(parents=True, exist_ok=True)  # make labels directory

    pbar = tqdm((dir / 'annotations').glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        src_img = cv2.imread(str((dir / 'images' / f.name).with_suffix('.jpg')))
        img_size = (src_img.shape[1], src_img.shape[0])
        with open(f, 'r') as file:  # read annotation.txt
            label_list = []
            lines = []
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                row = [int(x) for x in row]
                if row[4] == 0:  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                label_list.append([cls, row[0], row[1], row[2], row[3]])

            label_list = enhance_data(label_list, img_size, src_img)
            for lab in label_list:
                box = convert_box(img_size, tuple(map(int, lab[1:5])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
            label_path = str(f).replace(os.sep + 'annotations' + os.sep, os.sep + 'labels' + os.sep)
            label_path = label_path.replace(os.sep + 'VisDrone' + os.sep, os.sep + 'VisDronePro' + os.sep)
            with open(label_path, 'w') as fl:
                fl.writelines(lines)
        img_path = str((dir / 'images' / f.name).with_suffix('.jpg')).replace(os.sep + 'VisDrone' + os.sep,
                                                                              os.sep + 'VisDronePro' + os.sep)
        # print(label_path)
        # print(img_path)
        cv2.imwrite(img_path, src_img)


root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
dir = Path(root) / "datasets" / "VisDrone"
# visdrone2yolo(dir / "VisDrone2019-DET-train")
# print(cls_count)
# Convert
for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
    visdrone2yolo(dir / d)  # convert VisDrone annotations to YOLO labels
    pass
