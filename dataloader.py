from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
# from utils import *

# Data transformation with augmentation 
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        # 第一个参数表示图像的均值（mean），第二个表示标准差（standard deviation）
        # 这是ImageNet数据集的，其他数据集还需要修改
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Dataset
class LT_Dataset(Dataset):
    
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, path



# Load datasets
def load_data(data_root, dataset, phase, batch_size, num_workers=4, shuffle=True):
    txt = None
    if dataset == 'ImageNet_LT':
        if phase == 'train':
            txt = '/mnt/d/data/ImageNet_LT/ImageNet_LT_train.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/mnt/d/data/ImageNet_LT/ImageNet_LT_val.txt'
    
    elif dataset == 'CIFAR10_imb100':
        if phase == 'train':
            txt = '/mnt/d/data/cifar-10-batches-py/CIFAR10_train_imb100.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/mnt/d/data/cifar-10-batches-py/CIFAR10_val.txt'
    
    elif dataset == 'CIFAR10_imb50':
        if phase == 'train':
            txt = '/mnt/d/data/cifar-10-batches-py/CIFAR10_train_imb50.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/mnt/d/data/cifar-10-batches-py/CIFAR10_val.txt'
    
    elif dataset == 'CIFAR10_imb10':
        if phase == 'train':
            txt = '/mnt/d/data/cifar-10-batches-py/CIFAR10_train_imb10.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/mnt/d/data/cifar-10-batches-py/CIFAR10_val.txt'
    
    elif dataset == 'CIFAR100_imb100':
        if phase == 'train':
            txt = '/mnt/d/data/cifar-100-python/CIFAR100_train_imb100.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/mnt/d/data/cifar-100-python/CIFAR100_val.txt'
    
    elif dataset == 'CIFAR100_imb50':
        if phase == 'train':
            txt = '/mnt/d/data/cifar-100-python/CIFAR100_train_imb50.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/mnt/d/data/cifar-100-python/CIFAR100_val.txt'
    
    elif dataset == 'CIFAR100_imb10':
        if phase == 'train':
            txt = '/mnt/d/data/cifar-100-python/CIFAR100_train_imb10.txt'
        elif phase == 'val': #or phase == 'test':
            txt = '/mnt/d/data/cifar-100-python/CIFAR100_val.txt'
    else:
        print("数据集的txt没找到，再次检查问题。因加载数据集->",dataset)
    if txt is None:
        print("错了错了，txt怎么可能为None呢")
    print('Loading data from %s' % (txt))

    transform = data_transforms[phase]

    print('Use data transformation:', transform)

    set_ = LT_Dataset(data_root, txt, transform)
    print("加载数据集有 {} 条数据,来自 {}  ".format(len(set_),txt))


    print('Shuffle is %s.' % (shuffle))
    return DataLoader(dataset=set_, batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)
        
    
    
