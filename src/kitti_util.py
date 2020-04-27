import copy
from collections import Counter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.image as mpimg

from bbox import BBox2D, XYXY

def collate_fn(seq_list):
    img, target = zip(*seq_list)
    targets = [t for t in target]
    imgT = torch.stack([i for i in img])
    return (imgT, targets)

def get_iou(bb1, bb2):

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    if float(bb2_area) == 0.0:
        return 0.0
    return float(intersection_area)/float(bb2_area)

def inception_normalize(original_x):
    new_x = (0.299*original_x + 0.485) / 0.5 - 1
    return new_x

class KittiDataset(Dataset):
    def __init__(self, image_class_info):
        # TODO: maybe put the whole preprocess thing here?
        self.img_class_info = image_class_info
        #self.transform = transforms.Compose([transforms.ToTensor()])
                                             #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             #                     std=[0.229, 0.224, 0.225])])
        self.transform = transforms.Compose([transforms.ToTensor()])


    def __getitem__(self, index):
        row = self.img_class_info.iloc[index]
        img = cv2.imread(row["image_path"], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(img, (740, 224))

        # Cropping to 224x224 considering bounding box labels
        if row["class"] == 1:
            # Parse bboxes in image
            bbox_ls = [int(float(v)*(740/1240)) for v in row["bbox_l"].split(",")]
            bbox_ts = [int(float(v)*(224/375)) for v in row["bbox_t"].split(",")]
            bbox_rs = [int(float(v)*(740/1240)) for v in row["bbox_r"].split(",")]
            bbox_bs = [int(float(v)*(224/375)) for v in row["bbox_b"].split(",")]
            #print("Crop bound: [%.2f ~ %.2f]" % (min(bbox_ls), max(bbox_ls)))
            rand_box_idx = np.random.randint(0, len(bbox_ls))

            l_bnd, r_bnd = max(0, int(bbox_rs[rand_box_idx]-224)), min(img_resize.shape[1]-224, int(bbox_ls[rand_box_idx]))
            if l_bnd > r_bnd:
                l_bnd, r_bnd = r_bnd, l_bnd
            elif l_bnd == r_bnd:
                r_bnd += 1

            #print("Random cropping", (l_bnd, r_bnd))
            crop_idx = np.random.randint(l_bnd, r_bnd)
            #crop_idx = np.random.randint(min(img_resize.shape[1]-224, max(0, int(min(bbox_ls)))), min(int(max(bbox_ls)), img_resize.shape[1]-224)+1)
        else:
            crop_idx = np.random.randint(0, img_resize.shape[1]-224)
        img_crop = img_resize[:, crop_idx:crop_idx+224]
        assert(crop_idx >= 0)


        target = {}
        label = 1.0 if row["class"] == 1 else 0.0
        bboxs = []

        # Using bounding box info
        if row["class"] == 1:
            for (bbox_l, bbox_t, bbox_r, bbox_b) in zip(bbox_ls, bbox_ts, bbox_rs, bbox_bs):

                # Calculate coverage?
                crop_box = {"x1": crop_idx, "x2": crop_idx+224, "y1": 0, "y2": 224}
                obj_box = {"x1": bbox_l, "x2": bbox_r, "y1": bbox_t, "y2": bbox_b}
                # Positive tile if any of the pedestrian bounding box has a coverage of over 80%
                coverage = get_iou(crop_box, obj_box)
                bboxs.append(BBox2D((bbox_l-crop_idx, bbox_t, bbox_r-crop_idx, bbox_b), mode=XYXY))

        target["label"] = label
        target["bboxs"] = bboxs

        imgT = self.transform(img_crop)
        #print(imgT.shape, img_crop.shape)
        return (imgT, target)

    def __len__(self):
        return len(self.img_class_info)


def data_loader(args, test_path=False):

    balanced_image_class_info = pd.read_csv("/projectnb/saenkog/shawnlin/object-tracking/kitti-util/src/train_image_class_info.csv", sep=",")
    shuffled_image_class_info = balanced_image_class_info.sample(frac = 1.0)

    TRAIN_SIZE = int(len(shuffled_image_class_info) * 0.7)
    VAL_SIZE = int(len(shuffled_image_class_info) * 0.1)
    TEST_SIZE = int(len(shuffled_image_class_info) * 0.2)

    train_img_class_info = shuffled_image_class_info[:TRAIN_SIZE]
    val_img_class_info = shuffled_image_class_info[TRAIN_SIZE:TRAIN_SIZE+VAL_SIZE]
    test_img_class_info = shuffled_image_class_info[TRAIN_SIZE+VAL_SIZE:]

    print("Train size: %s" % len(train_img_class_info))
    print("Val size: %s" % len(val_img_class_info))
    print("Test size: %s" % len(test_img_class_info))

    tr_dataset = KittiDataset(train_img_class_info)
    tst_dataset = KittiDataset(test_img_class_info)
    val_dataset = KittiDataset(val_img_class_info)

    dataloaders = {
        "train": DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn),
        "val": DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn),
        "test": DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    }

    return dataloaders

if __name__ == "__main__":

    tr_ldr, val_ldr = data_loader(None)
    print(len(tr_ldr))
