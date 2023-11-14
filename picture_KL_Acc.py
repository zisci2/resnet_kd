import glob
import os

from matplotlib import pyplot as plt


def read_log_file(file_path):

    with open(file_path, 'r') as file:
        for line in file:
            if "Best val Acc" in line:
                values = line.strip().split(":")
                # print(values)
                acc = float(values[1].split(" ")[1])

                kl_mean = float(values[2].split(" ")[1])
                kl = float(values[2].split(" ")[2])

                Tkl_mean = float(values[3].split(" ")[1])
                Tkl = float(values[3].split(" ")[2])
                # print("acc:",acc,"kl_mean:",kl_mean,"kl:",kl,"Tkl_mean:",Tkl_mean,"Tkl:",Tkl)

        return acc,kl_mean,kl,Tkl_mean,Tkl

base_path = 'runs_student_01/'
# folders = [folder for folder in os.listdir(base_path)]
folders = [folder for folder in os.listdir(base_path) if "teacher8" in folder]

all_acc = []
all_kl_mean = []
all_kl = []
all_Tkl_mean = []
all_Tkl = []
for folder in folders:
    # txt_name = [name for name in os.listdir(os.path.join(base_path, folder)) if "txt" in name] 
    # log_file_path = os.path.join(base_path, folder, txt_name)
    # 使用glob模块来获取文件夹内符合条件的文件名, 更简洁
    log_file_path = glob.glob(os.path.join(base_path, folder, '*.txt'))
    if os.path.exists(log_file_path[0]):
        acc,kl_mean,kl,Tkl_mean,Tkl = read_log_file(log_file_path[0])

        all_acc.append(acc)
        all_kl_mean.append(kl_mean)
        all_kl.append(kl)
        all_Tkl_mean.append(Tkl_mean)
        all_Tkl.append(Tkl)

# 创建包含多个子图的图表
fig, axs = plt.subplots(2, 2, figsize=(10, 8))


# # x轴acc，y轴kl
# 绘制子图1: acc vs kl
axs[0, 0].plot(all_acc, all_kl, marker='o', linestyle='', label='Data Points')
axs[0, 0].set_xlabel('Accuracy')
axs[0, 0].set_ylabel('KL')
axs[0, 0].set_title('Accuracy vs. KL')
axs[0, 0].legend()

# 绘制子图2: acc vs kl_mean
axs[0, 1].plot(all_acc, all_kl_mean, marker='o', linestyle='', label='Data Points')
axs[0, 1].set_xlabel('Accuracy')
axs[0, 1].set_ylabel('KL Mean')
axs[0, 1].set_title('Accuracy vs. KL Mean')
axs[0, 1].legend()

# 绘制子图3: acc vs Tkl
axs[1, 0].plot(all_acc, all_Tkl, marker='o', linestyle='', label='Data Points')
axs[1, 0].set_xlabel('Accuracy')
axs[1, 0].set_ylabel('Tkl')
axs[1, 0].set_title('Accuracy vs. Tkl')
axs[1, 0].legend()

# 绘制子图4: acc vs Tkl_mean
axs[1, 1].plot(all_acc, all_Tkl_mean, marker='o', linestyle='', label='Data Points')
axs[1, 1].set_xlabel('Accuracy')
axs[1, 1].set_ylabel('Tkl Mean')
axs[1, 1].set_title('Accuracy vs. Tkl Mean')
axs[1, 1].legend()

########################################################
# # x轴kl，y轴acc
# # 绘制子图1: kl vs acc
# axs[0, 0].plot(all_kl, all_acc, marker='o', linestyle='', label='Data Points')
# axs[0, 0].set_xlabel('KL')
# axs[0, 0].set_ylabel('Accuracy')
# axs[0, 0].set_title('KL vs. Accuracy')
# axs[0, 0].legend()

# # 绘制子图2: kl_mean vs acc
# axs[0, 1].plot(all_kl_mean, all_acc, marker='o', linestyle='', label='Data Points')
# axs[0, 1].set_xlabel('KL Mean')
# axs[0, 1].set_ylabel('Accuracy')
# axs[0, 1].set_title('KL Mean vs. Accuracy')
# axs[0, 1].legend()

# # 绘制子图3: Tkl vs acc
# axs[1, 0].plot(all_Tkl, all_acc, marker='o', linestyle='', label='Data Points')
# axs[1, 0].set_xlabel('Tkl')
# axs[1, 0].set_ylabel('Accuracy')
# axs[1, 0].set_title('Tkl vs. Accuracy')
# axs[1, 0].legend()

# # 绘制子图4: Tkl_mean vs acc
# axs[1, 1].plot(all_Tkl_mean, all_acc, marker='o', linestyle='', label='Data Points')
# axs[1, 1].set_xlabel('Tkl Mean')
# axs[1, 1].set_ylabel('Accuracy')
# axs[1, 1].set_title('Tkl Mean vs. Accuracy')
# axs[1, 1].legend()
#######################################################


# 调整子图之间的布局
plt.tight_layout()

# # 显示图表
# plt.show()  # wsl 无法显示
# 保存图像
plt.savefig('plot.png')
