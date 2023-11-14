# resnet-kd

训练阶段: 测试代码能运行成功不
python main.py --mode='train' --arch='resnet50' --dataset="ImageNet_LT" --epochs=2
python main.py --mode='train' --arch='resnet18' --dataset="ImageNet_LT" --epochs=20

蒸馏
python main.py --mode='distil' --dataset="ImageNet_LT" --teacher_num=2 --epochs=15 --temp=5.0 --t_path 0 1
蒸馏效果不好可能是因为温度、教师性能等等，但这里只看KL对acc的影响，所以acc不一定是升高的，有点遗憾后续再改进吧，不然多教师比单教师好还真不好说明



imagenet上
预训练的resnet50：val Loss: 0.4505 Acc: 0.8720

runs_teacher/15-0.pt  Acc: 0.8437
runs_teacher/15-1.pt  Acc: 0.8354
runs_teacher/15-2.pt  Acc: 0.8311
runs_teacher/15-3.pt  Acc: 0.8275
runs_teacher/15-4.pt  Acc: 0.8249
runs_teacher/15-5.pt  Acc: 0.8220
runs_teacher/15-6.pt  Acc: 0.8234
runs_teacher/15-7.pt  Acc: 0.8243
runs_teacher/15-8.pt  Acc: 0.8250
runs_teacher/15-9.pt  Acc: 0.8257
runs_teacher/15-10.pt  Acc: 0.8254
runs_teacher/15-11.pt  Acc: 0.8249
runs_teacher/15-12.pt  Acc: 0.8243
runs_teacher/15-13.pt  Acc: 0.8253
runs_teacher/15-14.pt  Acc: 0.8250
