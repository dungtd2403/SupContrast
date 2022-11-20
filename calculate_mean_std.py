
import numpy as np

import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter



def calculate_mean_std(loader):
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        data = data[0]
        print(len(data))
        batch_samples = len(data)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

if __name__=="__main__":
    data_folder ="/home/dung/ZaloAI2022/ZaloAI_challenge/train/split_img"
    batch_size = 16
    train_sampler = None
    num_workers = 2
    train_transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor()    
    ])
    train_dataset = datasets.ImageFolder(root=data_folder,
                                            transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    mean,std =calculate_mean_std(train_loader)
    print(f'mean:{mean}')
    print(f'std:{std}')