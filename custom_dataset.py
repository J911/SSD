import torch
import torch.utils.data as data
import numpy as np

import glob
import os
import cv2

from lib.augmentations import preproc_for_test, preproc_for_train
from config import opt

class CustomDetection(data.Dataset):
    def __init__(self, root, dbtype='train'):
        self.root = root
        self.dbtype = dbtype
        self.label_dir = self.root + '/' + self.dbtype + '/'
        self.image_dir = self.root + '/images/' 

        self.labels = glob.glob(self.label_dir + '*')

    def __getitem__(self, index):
        anno_path = self.labels[index]
        img_path = self.image_dir + anno_path[len(self.label_dir):-4] + '.png' 

        bboxs, cls_ids = self.get_annotations(anno_path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)

        if self.dbtype == 'train':
            image, bboxs, cls_ids = preproc_for_train(image, bboxs, cls_ids, opt.min_size, opt.mean)
            image = torch.from_numpy(image)

        target = np.concatenate([bboxs, cls_ids.reshape(-1,1)], axis=1)

        return image, target

    def get_annotations(self, path):
        cls_ids = []
        bboxs = []  

        labels = open(path)
        line = labels.readline()

        while line:
            label = [float(x) for x in line.split(' ')]
            
            xmin = (label[1] * 800) - (label[3] * 800 / 2)
            ymin = (label[2] * 240) - (label[4] * 240 / 2)

            xmax = (label[1] * 800) + (label[3] * 800 / 2)
            ymax = (label[2] * 240) + (label[4] * 240 / 2)

            cls_id = int(label[0])
            bbox = [xmin, ymin, xmax, ymax]

            cls_ids.append(cls_id)
            bboxs.append(bbox)

            line = labels.readline()

        return np.array(bboxs), np.array(cls_ids)

    def __len__(self):
        return len(self.labels)
