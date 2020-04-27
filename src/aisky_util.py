import re
import os
import copy
from collections import Counter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from bbox import BBox2D, XYXY, XYWH
import ast

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.image as mpimg

def collate_fn(seq_list):
    targets = [t[0][1] for t in seq_list]
    imgT = torch.stack([t[0][0] for t in seq_list])
    return (imgT, targets)

trans_fn = transforms.Compose([transforms.ToTensor()])

def loader_fn(path):
    img_class = re.sub(r".*?/(has_car|no_car)/\d+\.png", "\\1", path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    imgT = trans_fn(img)
    target = {}
    label = 1.0 if img_class == "has_car" else 0.0
    target["bboxs"] = []
    target["label"] = label

    return imgT, target

def loader_bbox_fn(path):
    img_class = re.sub(r".*?/(has_car|no_car)/\d+\.png", "\\1", path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    imgT = trans_fn(img)
    target = {}
    label = 1.0 if img_class == "has_car" else 0.0
    target["label"] = label

    target["bboxs"]= []
    if label == 1.0:
        assert(path in bbox_map)
        target["bboxs"] = bbox_map[path][0]
    return imgT, target

def get_bbox_mapping(path):
    bbox_map = pd.read_csv(os.path.join(path, "bbox.csv"), sep=",")
    bbox_map["bbox"] = bbox_map["bbox"].apply(lambda x: parse_bboxs_from_str(x))
    bbox_map = bbox_map.set_index("has_car_img_paths").T.to_dict("list")
    return bbox_map

def parse_bboxs_from_str(s):
    "ex: [(23, 0, 154, 87), (75, 213, 195, 256), (158, 8, 256, 98)]"

    bbox_raw_list = ast.literal_eval(s)
    bbox_list = []
    K = 224/256
    for bbox in bbox_raw_list:
        bbox_list.append(to_bbox2d(bbox, norm_coords))
    return bbox_list

def norm_coords(v, K=224/256):
    return int(K*v)

def to_bbox2d(bbox, f):
    return BBox2D((f(bbox[0]), f(bbox[1]), f(bbox[2]-bbox[0]), f(bbox[3]-bbox[1])), mode=XYWH)

bbox_map = get_bbox_mapping("/projectnb/saenkog/awong1/dataset/aiskyeye/processed_iou0.1_256x256_2/testing/")

def data_loader(args, test_path=False):
    tr_dir = "/projectnb/saenkog/awong1/dataset/aiskyeye/processed_iou0.1_256x256_old/training"
    val_dir = "/projectnb/saenkog/awong1/dataset/aiskyeye/processed_iou0.1_256x256_old/validation"
    tst_dir = "/projectnb/saenkog/awong1/dataset/aiskyeye/processed_iou0.1_256x256_2/testing"

    tr_dataset = ImageFolder(root=tr_dir, loader=loader_fn)
    val_dataset = ImageFolder(root=tr_dir, loader=loader_fn)
    tst_dataset = ImageFolder(root=tst_dir, loader=loader_bbox_fn)
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    dataloaders = {
        "train": tr_loader,
        "val": val_loader,
        "test": tst_loader
    }

    return dataloaders

