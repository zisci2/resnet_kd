**# resnet-kd**



训练阶段:

python main.py --mode='train' --arch='resnet18' --dataset="ImageNet_LT" --epochs=20



蒸馏

python main.py --mode='distil' --dataset="ImageNet_LT" --teacher_num=2 --epochs=15 --temp=5.0 --t_path 0 1







imagenet上

预训练的resnet50：val Loss: 0.4505 Acc: 0.8720



runs_teacher/15-8.pt  Acc: 0.8250

runs_teacher/15-10.pt  Acc: 0.8254

