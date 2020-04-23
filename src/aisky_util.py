import re
import copy
from collections import Counter

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

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

def data_loader(args, test_path=False):
    tr_dir = "/projectnb/saenkog/awong1/dataset/aiskyeye/processed_iou0.05_256x256/training"
    val_dir = "/projectnb/saenkog/awong1/dataset/aiskyeye/processed_iou0.05_256x256/validation"
    tst_dir = "/projectnb/saenkog/awong1/dataset/aiskyeye/processed_iou0.05_256x256/testing"

    tr_dataset = ImageFolder(root=tr_dir, loader=loader_fn)
    val_dataset = ImageFolder(root=tr_dir, loader=loader_fn)
    tst_dataset = ImageFolder(root=tst_dir, loader=loader_fn)
    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)
    tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    dataloaders = {
        "train": tr_loader,
        "val": val_loader,
        "test": tst_loader
    }

    return dataloaders
