import copy
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import wasserstein_distance as w_distance

    
    
def loss_kd(outputs, teacher_outputs, labels, temp, alpha):
    beta = 1. - alpha
    q = F.log_softmax(outputs/temp, dim=1)
    p = F.softmax(teacher_outputs/temp, dim=1)
    soft_loss = nn.KLDivLoss(reduction='batchmean')(q, p) * temp ** 2 
    hard_loss = nn.CrossEntropyLoss()(outputs, labels)
    KD_loss = alpha * hard_loss + beta * soft_loss # 软硬损失的加权组合

    return KD_loss

def KL_divergence(model1_logits, model2_logits):
    # 计算的是两个模型输出的概率分布之间的 KL 散度  
    probs2 = F.softmax(model2_logits, dim=1)
    log_probs1 = F.log_softmax(model1_logits, dim=1, dtype=torch.double)
    kl_div = F.kl_div(log_probs1, probs2, reduction='batchmean', log_target=False) * 10 # 太小了,放大一些
    return kl_div.item()


def get_logfile_name(path):
    get_time = datetime.datetime.now().strftime('%b%d_%H-%M') # 月 日 时 分
    file_name = get_time + '_log.txt'
    
    if not os.path.exists(path):  
        os.makedirs(path)  
        
    return os.path.join(path, file_name)


def print_write(print_str, log_file):
    print(*print_str)
    with open(log_file, 'a') as f:
        print(*print_str, file=f)


def train_model(model, 
                dataloaders,
                optimizer, 
                scheduler, 
                tensorboard_writer,
                device,
                temp,
                alpha,
                log_file,
                teacher=None,
                num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_acc = 0.0
    best_KL_mean = 0.0
    best_Tkl_mean = 0.0
    dataset_sizes = {x: len(dataloaders[x].dataset)
                     for x in ['train', 'val']}
        
    for epoch in range(num_epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            results_dict = {
                # 两教师之间的Kl
                'trian_kl': 0.0, 
                'val_kl': 0.0,
                'batch_kl': 0.0,
                # 单教师与两教师平均的KL，单教师和多教师的区别
                '2Tand1T_trian_kl': 0.0, 
                '2Tand1T_val_kl': 0.0,
                '2Tand1T_batch_kl': 0.0, 
                # 单教师与自己的kl
                '1T1T_trian_kl': 0.0, 
                '1T1T_val_kl': 0.0,
                '1T1T_batch_kl': 0.0, 
                            }
            
            for batch_idx, (inputs, labels, _) in enumerate(tqdm(dataloaders[phase])):
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.cuda.amp.autocast():
                    teacher_outputs = []
                    KD_loss = []
                    with torch.no_grad():
                        # if teacher and phase == 'train': # 若是蒸馏情况有教师存在，则不用更新梯度
                        if teacher: # 若是蒸馏情况有教师存在，则不用更新梯度
                            # 得到教师的logits
                            for i in range(len(teacher)):
                                teacher[i] = teacher[i].to(device)
                                teacher_outputs.append(teacher[i](inputs)) 
                            # if torch.equal(teacher_outputs[0], teacher_outputs[1]):
                            #     print("teacher_outputs是一样的")
                            # else:
                            #     print("teacher_outputs是bu一样的")

                            if len(teacher) == 2 and phase == 'train':
                                # 1T 与 2T 做KL 来看看KL大的时候acc怎么样,小的时候又怎么样
                                kl_div = KL_divergence(teacher_outputs[0], (teacher_outputs[0]+teacher_outputs[1])/2)
                                results_dict['2Tand1T_batch_kl'] = kl_div
                                results_dict['2Tand1T_trian_kl'] += kl_div

                                # 1T 与 1T 做KL 来看看KL大的时候acc怎么样,小的时候又怎么样-->应该为0吧
                                kl_div = KL_divergence(teacher_outputs[0], (teacher_outputs[0]+teacher_outputs[0])/2)
                                results_dict['1T1T_batch_kl'] = kl_div
                                results_dict['1T1T_trian_kl'] += kl_div

                                kl_div = KL_divergence(teacher_outputs[0], teacher_outputs[1])
                                results_dict['batch_kl'] = kl_div
                                results_dict['trian_kl'] += kl_div

                            if len(teacher) == 2 and phase == 'val':
                                # 1T 与 2T 做KL 来看看KL大的时候acc怎么样,小的时候又怎么样
                                kl_div = KL_divergence(teacher_outputs[0], (teacher_outputs[0]+teacher_outputs[1])/2)
                                results_dict['2Tand1T_batch_kl'] = kl_div
                                results_dict['2Tand1T_val_kl'] += kl_div

                                # 1T 与 1T 做KL 来看看KL大的时候acc怎么样,小的时候又怎么样-->应该为0吧
                                kl_div = KL_divergence(teacher_outputs[0], (teacher_outputs[0]+teacher_outputs[0])/2)
                                results_dict['1T1T_batch_kl'] = kl_div
                                results_dict['1T1T_trian_kl'] += kl_div

                                kl_div = KL_divergence(teacher_outputs[0], teacher_outputs[1])
                                results_dict['batch_kl'] = kl_div
                                results_dict['val_kl'] += kl_div

                                
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)

                        if teacher and phase == 'train': 
                            criterion = loss_kd
                            for i in range(len(teacher)):
                                KD_loss.append(criterion(outputs, teacher_outputs[i], labels, temp, alpha))
                            loss = sum(KD_loss) / len(KD_loss)
                            # if KD_loss[0] == KD_loss[1]:
                            #     print("KD_loss是一样的")
                            # else:
                            #     print("KD_loss是bu一样的")
                        else: # 无教师或者学生模型进行val的时候
                            criterion = nn.CrossEntropyLoss()
                            loss = criterion(outputs, labels)

                        if phase == 'train': # 这个阶段不是mode=['distil','train']
                            loss.backward()
                            optimizer.step()  #TODO 可以比较一下参数，看是否改变

                batch_acc = (torch.sum(preds == labels.data) / len(labels))
                if len(teacher) == 2:
                    # 现在画的是batch内的
                    # 使用tensoboard记录多个值
                    if phase == 'train':
                        tensorboard_writer.add_scalars('batch/train', 
                                                    {'TKL':results_dict['2Tand1T_batch_kl'],'sameTKL':results_dict['1T1T_batch_kl'],'KL':results_dict['batch_kl'],'acc':batch_acc}, 
                                                    epoch * len(labels) + batch_idx)
                        tensorboard_writer.add_scalars('batch/train_kl', 
                                                    {'KL':results_dict['batch_kl'],'acc':batch_acc}, 
                                                    epoch * len(labels) + batch_idx)
                        tensorboard_writer.add_scalars('batch/train_Tkl', 
                                                    {'TKL':results_dict['2Tand1T_batch_kl'],'acc':batch_acc}, 
                                                    epoch * len(labels) + batch_idx)
                    else: # val
                        tensorboard_writer.add_scalars('batch/val', 
                                                    {'TKL':results_dict['2Tand1T_batch_kl'],'sameTKL':results_dict['1T1T_batch_kl'],'KL':results_dict['batch_kl'],'acc':batch_acc}, 
                                                    epoch * len(labels) + batch_idx)
                        tensorboard_writer.add_scalars('batch/val_kl', 
                                                    {'KL':results_dict['batch_kl'],'acc':batch_acc}, 
                                                    epoch * len(labels) + batch_idx)
                        tensorboard_writer.add_scalars('batch/val_Tkl', 
                                                    {'TKL':results_dict['2Tand1T_batch_kl'],'acc':batch_acc}, 
                                                    epoch * len(labels) + batch_idx)
                        
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if  phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (running_corrects.double() / dataset_sizes[phase])
            print_str = [f'Epoch_{epoch+1}/{num_epochs} {phase}_Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}']
            print_write(print_str, log_file)


            # 值太大了，取个平均
            results_dict['1T1T_trian_kl'] = results_dict['1T1T_trian_kl'] / dataset_sizes[phase]
            results_dict['1T1T_val_kl'] = results_dict['1T1T_val_kl'] / dataset_sizes[phase]

            T2and1T_trian_kl = results_dict['2Tand1T_trian_kl']
            T2and1T_val_kl = results_dict['2Tand1T_val_kl']
            results_dict['2Tand1T_trian_kl'] = results_dict['2Tand1T_trian_kl'] / dataset_sizes[phase]
            results_dict['2Tand1T_val_kl'] = results_dict['2Tand1T_val_kl'] / dataset_sizes[phase]

            trian_kl = results_dict['trian_kl']
            val_kl = results_dict['val_kl']
            results_dict['trian_kl'] = results_dict['trian_kl'] / dataset_sizes[phase]
            results_dict['val_kl'] = results_dict['val_kl'] / dataset_sizes[phase]
            if len(teacher) == 2:
                # 现在画的是epoch内的
                if phase == 'train':
                    tensorboard_writer.add_scalars('train/1T2T', 
                                                {'TKL':results_dict['2Tand1T_trian_kl'],'sameTKL':results_dict['1T1T_trian_kl'],'KL':results_dict['trian_kl'],'acc':epoch_acc}, 
                                                epoch)
                else: # val
                    tensorboard_writer.add_scalars('val/1T2T', 
                                                {'TKL':results_dict['2Tand1T_val_kl'],'sameTKL':results_dict['1T1T_val_kl'],'KL':results_dict['val_kl'],'acc':epoch_acc}, 
                                                epoch)            

            # 训练的loss和acc
            if phase == "train":
                tensorboard_writer.add_scalar("train/loss",epoch_loss,epoch)
                tensorboard_writer.add_scalar("train/acc",epoch_acc,epoch)
            else: # val 
                tensorboard_writer.add_scalar("val/loss",epoch_loss,epoch)
                tensorboard_writer.add_scalar("val/acc",epoch_acc,epoch)


            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_KL_mean = results_dict['val_kl']
                best_KL = val_kl
                best_Tkl_mean = results_dict['2Tand1T_val_kl']
                best_Tkl = T2and1T_val_kl
                best_model_wts = copy.deepcopy(model.state_dict())
            
            # # 某个点保存一下
            # if epoch % 20 == 0:
            #     save_path = f"runs/{dataset}_{mode}_{arch}_{epochs}/{dataset}_{mode}_{arch}_{epochs}.pt"
            #     torch.save(model, save_path)

    
    time_elapsed = time.time() - since  
    hours = time_elapsed // 3600  
    minutes = (time_elapsed % 3600) // 60  
    seconds = time_elapsed % 60  
    print_str = [f'Training complete in {hours:.0f}h {minutes:.0f}m {seconds:.0f}s\n',  
                # f'Best val Acc: {best_acc:.4f}',  # 不应该限制小数的，比较差距这么小万一都相同了怎么办
                f"Best validation accuracy is {best_acc} at epoch {best_epoch}"
                f"KL: {best_KL_mean} {best_KL}",
                f"TKL: {best_Tkl_mean} {best_Tkl}"]
    print_write(print_str, log_file)
    model.load_state_dict(best_model_wts)

    return model