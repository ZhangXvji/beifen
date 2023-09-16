import torch
import numpy as np
import multi_swin
from dataset import DogDataset
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data_simulator import CreateDataset
import os

class Args:
    def __init__(self) -> None:
        self.epochs, self.learning_rate, self.patience = [50, 1e-3, 5]
        self.batch_size = 16
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        # tensorboard的log
        self.logs = "./modal_fusion/logs/train_log"
        self.video_frames = 10
        self.pretrained = './model/swin_tiny_patch244_window877_kinetics400_1k.pth'


class EarlyStopping():
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },'./model/model_checkpoint.pth')
        self.val_loss_min = val_loss

def calculate_accuracy(output,label):
    # 将模型输出转换为类别预测
    predicted_classes = torch.argmax(output, dim=2)  # 在第三维上找到最大值的索引

    # 将标签转换为类别
    true_classes = torch.argmax(label, dim=2)  # 在第三维上找到最大值的索引

    # 计算准确率
    correct_predictions = torch.sum(predicted_classes == true_classes)  # 统计正确预测的数量
    total_predictions = label.size(0) * label.size(1)  # 总共的预测数量，batch_size * 序列长度

    accuracy = correct_predictions.item() / total_predictions
    print("Accuracy:", accuracy)
    return accuracy




args = Args()
writer = SummaryWriter(log_dir=args.logs)

# bev = torch.rand(64, 1, 60, 200, 200)
# video = torch.rand(64, 3, 60, 56, 56)
# imu = torch.rand(64, 1, 60, 10)
# sensor = torch.rand(64, 1, 60, 4)
# motor = torch.rand(64, 1, 60, 2)
# label = torch.rand(64, 20)
dataset = DogDataset()
total_sample = len(dataset)
train_size = int(0.8*total_sample)
test_size = total_sample - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)


# ======================train============================

model = multi_swin.MultiSwin(
    input_channels=1, output_classes=128, final_output_classes=10, pretrained=args.pretrained, debug=False).to(
    args.device)
# model = nn.DataParallel(model) # 使用全部GPU

# device_ids = [0] 
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# model = torch.nn.DataParallel(model, device_ids=device_ids)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)

# optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []

early_stopping = EarlyStopping(patience=args.patience, verbose=True)

# 检查是否有预训练的模型文件，如果有，则加载
checkpoint_path = "./model/model_checkpoint.pth"  # 指定模型 checkpoint 文件的路径
if(os.path.exists(checkpoint_path)):
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        # 如果不在 GPU 上训练，需要将模型加载到 CPU 上
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


for epoch in range(args.epochs):
    model.train()
    train_epoch_loss = []
    for idx, sample_batch in enumerate(train_dataloader, 0):
        optimizer.zero_grad()
        outputs = model(sample_batch['bev'].to(args.device), sample_batch['video'].to(args.device),
                        sample_batch['imu'].to(args.device), sample_batch['sensor'].to(args.device),
                        sample_batch['motor'].to(args.device))
        loss = criterion(outputs, (sample_batch['label']).to(args.device))
        loss.backward()
        optimizer.step()

        train_epoch_loss.append(loss.item())
        train_loss.append(loss.item())
        # if idx % (len(test_dataloader)//2) == 0:
        print("--epoch={}/{}    {}/{} of train     loss={}".format(
            epoch, args.epochs, idx, len(train_dataloader), loss.item()))
        # writer.add_video('video', sample_batch['video'], epoch * len(train_dataloader) + idx)
        # writer.add_images('rgb', sample_batch['rgb'], epoch * len(test_dataloader) + idx, dataformats='NCHW')
        accuracy = calculate_accuracy(outputs,sample_batch['label'].to(args.device))  # 用于计算准确率的函数
        writer.add_scalar('train_accuracy', accuracy, epoch * len(train_dataloader) + idx)
        writer.add_scalar('train_loss', loss.item(), epoch * len(train_dataloader) + idx)
    train_epochs_loss.append(np.average(train_epoch_loss))

    # =====================valid============================
    with torch.no_grad():
        model.eval()
        valid_epoch_loss = []
        for idx, sample_batch in enumerate(test_dataloader, 0):
            outputs = model(sample_batch['bev'].to(args.device), sample_batch['video'].to(args.device),
                        sample_batch['imu'].to(args.device), sample_batch['sensor'].to(args.device),
                        sample_batch['motor'].to(args.device))

            loss = criterion(outputs, sample_batch['label'].to(args.device))
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
            # if idx % (len(test_dataloader)//2) == 0:
            print("--epoch={}/{}    {}/{} of val     loss={}".format(
                epoch, args.epochs, idx, len(test_dataloader), loss.item()))
            # writer.add_video('video', sample_batch['video'], epoch * len(test_dataloader) + idx)
            # writer.add_image('rgb', sample_batch['rgb'], epoch * len(test_dataloader) + idx, dataformats='NCHW')
            accuracy = calculate_accuracy(outputs,sample_batch['label'].to(args.device))  # 用于计算准确率的函数
            writer.add_scalar('valid_accuracy', accuracy, epoch * len(test_dataloader) + idx)
            writer.add_scalar('valid_loss', loss.item(), epoch * len(test_dataloader) + idx)
        valid_epochs_loss.append(np.average(valid_epoch_loss))
    # ==================early stopping======================
    early_stopping(
        valid_epochs_loss[-1], model=model, path=r'./model')
    if early_stopping.early_stop:
        print("Early stopping")
        break
    # ====================adjust lr========================
    lr_adjust = {
        5: 5e-4,
        10: 1e-4,
        15: 5e-5,
        20: 1e-5,
        30: 5e-6,
        45: 1e-6,
        40: 5e-7,
    }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

# =========================save model=====================
torch.save(model, './model.pth')
writer.close()

# plt.figure(figsize=(12, 4))
# plt.subplot(121)
# plt.plot(train_loss[:])
# plt.title("train_loss")
# plt.subplot(122)
# plt.plot(train_epochs_loss[1:], '-o', label="train_loss")
# plt.plot(valid_epochs_loss[1:], '-o', label="valid_loss")
# plt.title("epochs_loss")
# plt.legend()
# plt.show()