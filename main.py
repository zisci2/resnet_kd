import argparse

import torch
import torch.autograd.profiler as profiler
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models
from torchvision import transforms as T

from trainer import loss_kd, train_model,get_logfile_name,print_write
from dataloader import load_data

#TODO 需要修改的地方 
# 数据集加载 (✔)
# 单专家->多教师蒸馏 (✔)
# 记录KL距离 和 wassertein距离 (✔)
# 多教师距离近还是远？--> 这个不懂这么画啊
# 记录KL到txt文件中 --> 没记,主要不知道那些是有用的

resnets = ['resnet18', 'resnet34', 'resnet50']
students = ['resnet18', 'resnet34']
modes = ['train', 'distil'] 
data_root = {'CIFAR100': '/mnt/d/data/cifar-100-python/clean_img',
             'CIFAR10': '/mnt/d/data/cifar-10-batches-py/clean_img',
             'ImageNet':'/mnt/e/dataset/ImageNet/data/ImageNet2012'}
teacher_path = [
    # "./runs_teacher/15-0.pt",
    # "./runs_teacher/15-1.pt",
    # "./runs_teacher/15-2.pt",
    # "./runs_teacher/15-3.pt",
    # "./runs_teacher/15-4.pt",
    # "./runs_teacher/15-5.pt",
    # "./runs_teacher/15-6.pt",
    # "./runs_teacher/15-7.pt",
    "./runs_teacher/15-8.pt",
    # "./runs_teacher/15-9.pt",
    "./runs_teacher/15-10.pt",
    # "./runs_teacher/15-11.pt",
    # "./runs_teacher/15-12.pt",
    # "./runs_teacher/15-13.pt",
    # "./runs_teacher/15-14.pt",
]

parser = argparse.ArgumentParser(description='Distillation from resnet50')
parser.add_argument('--mode', default='distil', choices=modes,
                    help='program mode: ' +
                        ' | '.join(resnets) +
                        ' (default: train)')
parser.add_argument('--arch', default='resnet18', choices=resnets,
                    help='model architecture: ' +
                        ' | '.join(resnets) +
                        ' (default: resnet18)')
parser.add_argument('--teacher_num', default=0, type=int,
                    help='教师模型的数量')
parser.add_argument('--t_path', default=[0,1], nargs='+', type=int,
                    help='教师模型的路径选择')
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=256, type=int,
                    help='批次大小')
parser.add_argument('--dataset', default="ImageNet_LT", # eg.CIFAR100_imb100
                    help='确定用的是什么数据集，不平衡比是什么样的，一律以下划线连接')
parser.add_argument('-t', '--temp', default=10., type=float,
                    help='temperature for distillation')
parser.add_argument('--alpha', default=0.2, type=float,
                    help='weighting for hard loss during distillation')


def compare_models(model1, model2):
    # 获取两个模型的状态字典
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()

    # 检查模型参数的键是否相同
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    if keys1 != keys2:
        print("teacher模型参数的键不同")
        return False
    # 检查每个参数是否相同
    for key in keys1:
        if not torch.equal(state_dict1[key], state_dict2[key]):
            print(f"teacher模型参数 {key} 不同")
            return False

    print("两teacher模型参数一致")
    return True

def train(args, path, log_file, device):

    model = models.__dict__[args.arch](weights=None)
    # model = models.__dict__[arch](pretrained=False) # 学生就不用预训练了

    data = {x: load_data(data_root=data_root[args.dataset.split("_")[0]], dataset=args.dataset, phase=x,
                                batch_size=args.batch_size, num_workers=4,
                                shuffle=True)
        for x in ['train', 'val']}     
    
    if "CIFAR10_" in args.dataset:
        class_num = 10
    elif "CIFAR100_" in args.dataset:
        class_num = 100
    elif "ImageNet" in args.dataset:
        class_num = 1000
    else:
        class_num = None
        print("错了再来，没设置样本总类别数")

    model_ft = model
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, class_num)
    # if args.arch == 'resnet50' and args.mode == 'train':
    #     print("加载resnet50预训练模型了，注意了啊")
    #     model_ft.load_state_dict(torch.load("./resnet50.pt"))
    model_ft = model_ft.to(device)

    if args.mode == 'distil':
        teacher = []
        for i in range(args.teacher_num):
            teacher_model = models.resnet50(pretrained=False) 
            print("加载的教师模型是：",teacher_path[args.t_path[i]])
            teacher_model.load_state_dict(torch.load(teacher_path[args.t_path[i]])) 
            teacher.append(teacher_model)
        compare_models(teacher[0], teacher[1])
    else:
        teacher = None

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    print_str = [f"--teacher: {f'resnet50' if teacher else 'None'}",f"--teacher number: {args.teacher_num}\n"
                 f"--teacher_1: {teacher_path[args.t_path[0]]}",f"--teacher_2: {teacher_path[args.t_path[1]]}\n"
                 f"--dataset: {args.dataset}",f"--epochs: {args.epochs}",f"--batch size: {args.batch_size}\n"
                 f"--distillation temperature: {args.temp}\n"
                 f"--hard label weight: {args.alpha}",f"--soft label weight: {1. - args.alpha}\n"]
    print_write(print_str, log_file)
    writer = SummaryWriter(path)
    model_ft = train_model(model_ft, data, optimizer_ft, 
                           exp_lr_scheduler, writer, device, args.temp, args.alpha, log_file,
                           teacher, args.epochs)
    save_path = f"{path}/{args.mode}_{args.arch}_{args.epochs}.pt"
    torch.save(model_ft.state_dict(), save_path)


def main():
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f'Mode: {args.mode}')
    if args.mode == 'train':
        print_str = [f'Training model: {args.arch}...']
        path = f'runs/{args.dataset}_{args.mode}_{args.epochs}'
        log_file = get_logfile_name(path=path)
        print_write(print_str, log_file)
        train(args, path, log_file, device)
    elif args.mode == 'distil':
        print_str = [f'Training student: {args.arch}...']
        path = f'runs/{args.dataset}_{args.mode}_teacher{args.t_path[0]}-{args.t_path[1]}_{args.epochs}'
        log_file = get_logfile_name(path=path)
        print_write(print_str, log_file)
        train(args, path, log_file, device)


if __name__ == '__main__':
    main()
