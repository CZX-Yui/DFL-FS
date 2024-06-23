import torch

import numpy as np
import copy
import random

from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data.dataset import Dataset

def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(int(idx))
    count_len = []
    for i in range(num_classes):
        count_len.append(len(list1[i]))
    return list1, count_len


# 联邦式数据分布划分
def CIFAR10_client_longtail(dataset_raw, num_users, imb_ratio, 
                            head_ratio, local_imb=False):
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    indices_per_class, _ = classify_label(dataset_raw, num_classes=10)
    num_per_class = []
    for i in range(10):
        head_num = head_ratio * len(indices_per_class[i]) / num_users 
        class_num_i = round(head_num * (imb_ratio ** (i / (10 - 1.0))))
        num_per_class.append(class_num_i)

    for class_i in range(10):
        np.random.shuffle(indices_per_class[class_i])

    if not local_imb:
        for client_i in range(num_users):
            for class_i in range(10):
                choose_num_i = num_per_class[class_i]
                choose_indices_i = indices_per_class[class_i][client_i*choose_num_i : (client_i+1)*choose_num_i]
                dict_users[client_i] = np.concatenate((dict_users[client_i], choose_indices_i), axis=0, dtype=int)
    else:
        for client_i in range(num_users):
            shift = random.randint(0,25)
            num_per_class_i = num_per_class[shift:] + num_per_class[:shift]
            for class_i in range(26):
                choose_num_i = num_per_class_i[class_i]
                choose_indices_i = np.random.choice(indices_per_class[class_i], choose_num_i, replace=False)
                dict_users[client_i] = np.concatenate((dict_users[client_i], choose_indices_i), axis=0, dtype=int)

    return dict_users

def CIFAR10_check_LT_FL(dataset_raw, dict_uesr, num_client):
    weights = []
    for client_i in range(num_client):
        count = [0] * 10
        for idx in dict_uesr[client_i]:
            label_i = dataset_raw[int(idx)][1]
            count[label_i] += 1

        print(count)  
        sum_i = sum([1/i for i in count])
        weight = [(1/i)/sum_i for i in count]
        weight = torch.tensor(weight)

        weights.append(weight)
    return weights


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs=[]):
        self.dataset = dataset
        if isinstance(idxs, list):
            self.idxs = []
            for idx_i in idxs:        
                self.idxs.append(idx_i)
        else:
            self.idxs = idxs

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label
        

class Bal_DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.labels = dataset.targets

        final_idx = []

        check_label = [0] * 10       
        label_per_class = [[] for _ in range(10)]
        for i in idxs:
            check_label[self.labels[i]] += 1
            label_per_class[self.labels[i]].append(i)
        sample_per_class = int(sum(check_label) / 10)
        for c_i in range(10):
            if len(label_per_class[c_i]) > 0:
                _label_per_class = np.random.choice(label_per_class[c_i], sample_per_class)
                final_idx.extend(_label_per_class)

        self.idxs = final_idx

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


