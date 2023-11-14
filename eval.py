import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, utils
from torchvision import transforms as T
from tqdm import tqdm

from dataloader import load_data


def evaluate(model, dataset, device):
    model.eval()
    model.to(device)
    dataset_sizes = len(dataset.dataset)
    running_corrects = 0
    for i, (inputs, labels, _) in enumerate(tqdm(dataset)):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                running_corrects += torch.sum(preds == labels.data)

    acc = running_corrects.double() / dataset_sizes
    print("val 正确样本数-->",running_corrects)
    print("val 总样本数-->",dataset_sizes)
    print(f'Acc: {acc:.4f}')


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_root = {'CIFAR100': '/mnt/d/data/cifar-100-python/clean_img',
                'CIFAR10': '/mnt/d/data/cifar-10-batches-py/clean_img',
                'ImageNet':'/mnt/e/dataset/ImageNet/data/ImageNet2012'}
    dataset = "ImageNet_LT"
    batch_size = 128
    data = {x: load_data(data_root=data_root[dataset.split("_")[0]], dataset=dataset, phase=x,
                                batch_size=batch_size, num_workers=4,
                                shuffle=None)
        for x in ['val']} 
    
    if "CIFAR10_" in dataset:
        class_num = 10
    elif "CIFAR100_" in dataset:
        class_num = 100
    elif "ImageNet" in dataset:
        class_num = 1000
    else:
        class_num = None
        print("错了再来，没设置样本总类别数")

    # model = models.resnet50(pretrained=False)
    model = models.resnet50(weights=None)
    # model = models.resnet18(weights=None)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, class_num)
    # model.load_state_dict(torch.load("./resnet50.pt"))
    model.load_state_dict(torch.load("./runs_teacher/15-1.pt"))

    evaluate(model,data['val'],device)