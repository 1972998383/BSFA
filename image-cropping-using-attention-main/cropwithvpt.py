import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.optim as optim  # 导入 PyTorch 的优化器模块
import torchvision.transforms as transforms  # 导入 torchvision 的图像变换模块
import torchvision.datasets as datasets  # 导入 torchvision 数据集
from torch.utils.data import DataLoader  # 导入 PyTorch 的数据加载模块
import numpy as np  # 导入 Numpy 库
import cv2  # 导入 OpenCV 库，用于图像处理
import timm  # 导入 timm 库，用于加载预训练模型
# from models.net import VPTModel
from models.net import Trainer
from src.models.vit_models import ViT

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import datetime
import time
import warnings

import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from tqdm import tqdm
# from models.net import Net


from data_manager import DataManager
from utils.losses import CrossEntropyLoss
from utils.optimizers import init_optimizer

from utils.iotools import save_checkpoint
from utils import AverageMeter
from utils import Logger
from utils.torchtools import one_hot, adjust_learning_rate


from src.configs.config import get_cfg
from setup import cfg

def main(args):
    np.random.seed(args.seed)  # 设置随机种子
    torch.manual_seed(args.seed)  # 设置PyTorch随机种子
    torch.cuda.manual_seed_all(args.seed)  # 设置所有GPU随机种子
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices  # 设置可见的CUDA设备
    use_gpu = torch.cuda.is_available()  # 检查是否有GPU可用
    print("Currently using GPU {}".format(args.gpu_devices))  # 打印当前使用的GPU设备
    cudnn.benchmark = False  # 禁用cudnn加速
    cudnn.deterministic = True  # 设置cudnn确定性模式

    args.save_dir = os.path.join(args.save_dir, args.dataset, args.model,
                                 'global_' + str(args.weight_global) + '_local_' + str(args.weight_local))  # 设置保存目录

    # 数据预处理
    Dataset = DataManager(args, use_gpu)  # 数据管理器
    trainloader, testloader = Dataset.return_dataloaders()

    # pretrained_model_path = args.model # 加载预训练的 ViT 模型 (使用 timm 库)
    # pretrained_model = timm.create_model('vit_base_patch16_224', pretrained=True) # 创建 ViT 模型结构，设置 pretrained=False，因为我们手动加载预训练权重
    # feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    # vit_model = ViTForImageClassification.from_pretrained(model_name, num_labels=10)  # 10分类任务
    model = ViT(cfg=cfg, load_pretrain=True, vis=True)
    print(model)
    # base_model.load_state_dict(torch.load(pretrained_model_path)) # 加载预训练的权重
    print("Pretrained model loaded successfully.")

    # 初始化 Prompt Embedding
    # prompter = PromptedTransformer(num_prompts=5, embed_dim=base_model.embed_dim)


    if use_gpu:
        model = model.cuda()  # 将模型移动到GPU上

    optimizer = optim.Adam(model.prompt_embedding.parameters(), lr=1e-4)  # 仅优化 Prompt Embedding
    criterion1 = CrossEntropyLoss().cuda()  # 定义交叉熵损失函数
    criterion2 = torch.nn.CrossEntropyLoss().cuda()  # 定义交叉熵损失函数

    start_time = time.time()  # 记录开始时间
    train_time = 0  # 训练时间，初始化为0
    best_acc = -np.inf  # 最佳准确率，初始化为负无穷
    best_epoch = 0  # 最佳时期，初始化为0


    args.LUT_lr = [(60, 0.1), (70, 0.006), (80, 0.0012), (90, 0.00024)]

    for epoch in range(args.max_epoch):
        learning_rate = adjust_learning_rate(optimizer, epoch, args.LUT_lr)  # 调整学习率

        start_train_time = time.time()
        train(args, epoch, model, criterion1, criterion2, optimizer, trainloader, learning_rate, use_gpu)  # 训练模型
        train_time += round(time.time() - start_train_time)

        if epoch == 0 or epoch % 5 == 0 or epoch >= (args.LUT_lr[0][0] - 1):

            acc = test(model, testloader, use_gpu)  # 测试模型
            is_best = acc > best_acc

            if is_best:
                best_acc = acc
                best_epoch = epoch + 1

                save_checkpoint({
                    'state_dict': model.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.pth.tar'))  # 保存最佳模型

            print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))  # 打印最佳准确率

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))  # 打印总训练时间
    print("==========\nArgs:{}\n==========".format(args))  # 打印参数信息


def train(args, epoch, model, criterion1, criterion2, optimizer, trainloader, learning_rate, use_gpu):
    losses = AverageMeter()  # 平均损失
    batch_time = AverageMeter()  # 平均批处理时间
    data_time = AverageMeter()  # 数据加载时间

    model.train()  # 设置模型为训练模式

    end = time.time()  # 记录当前时间
    for batch_idx, (images_train, labels_train, images_test, labels_test, pids_1, pids_2) in enumerate(
            tqdm(trainloader)):  # 遍历训练数据加载器，获取每个批次的训练图像、训练标签、测试图像、测试标签、pids_1和pids_2，并使用tqdm显示进度条#pids_1和pids_2从数据集所取的类别
        # 对图像进行自动裁剪（根据 Attention 或边缘检测）

        data_time.update(time.time() - end)  # 更新数据加载时间

        if use_gpu:
            images_train, labels_train = images_train.cuda(), labels_train.cuda()  # 将训练图像和标签移至GPU
            images_test, labels_test = images_test.cuda(), labels_test.cuda()  # 将测试图像和标签移至GPU
            pids_1 = pids_1.cuda()  # 将pids_1移至GPU
            pids_2 = pids_2.cuda()  # 将pids_2移至GP

        pids = torch.cat((pids_1, pids_2), dim=1)  # 拼接pids_1和pids_2
        labels_train_1hot = one_hot(labels_train).cuda()  # 对训练标签进行one-hot编码并移至GPU
        labels_test_1hot = one_hot(labels_test).cuda()  # 对测试标签进行one-hot编码并移至GPU

        optimizer.zero_grad()  # 梯度清零
        trainer = Trainer()
        s1, s2, glo1, glo2 = trainer.duenei(model, images_train, labels_train, images_test, labels_test)  # 使用模型进行前向传播


        loss_global1 = criterion2(glo1, pids.view(-1))  # 计算全局损失1
        loss_global2 = criterion2(glo2, pids.view(-1))  # 计算全局损失2
        loss_global = 0.5 * loss_global1 + 0.5 * loss_global2  # 计算总的全局损失

        loss_xcos1 = criterion1(s1, labels_test.view(-1))  # 计算局部损失1
        loss_xcos2 = criterion1(s2, labels_test.view(-1))  # 计算局部损失2
        loss_xcos = 0.5 * loss_xcos1 + 0.5 * loss_xcos2  # 计算总的局部损失

        loss = loss_global * args.weight_global + loss_xcos * args.weight_local  # 计算总损失

        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        losses.update(loss.item(), pids_2.size(0))  # 更新损失值
        batch_time.update(time.time() - end)  # 更新批处理时间
        end = time.time()  # 记录当前时间

    tm = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    print('[{0}] '
          'Epoch{1} '
          'lr: {2} '
          'Time:{batch_time.sum:.1f}s '
          'Data:{data_time.sum:.1f}s '
          'Loss:{loss.avg:.4f} '.format(tm,
                                        epoch + 1, learning_rate, batch_time=batch_time,
                                        data_time=data_time, loss=losses))

def test(model, testloader, use_gpu):
    accs = AverageMeter()  # 用于计算准确率
    test_accuracies = []  # 存储每个batch的准确率
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images_train, labels_train, images_test, labels_test) in enumerate(tqdm(testloader)):  # 遍历测试数据集
            if use_gpu:  # 如果使用GPU
                images_train = images_train.cuda()
                images_test = images_test.cuda()

            q_batch_size, num_test_examples = images_test.size(0), images_test.size(1)

            labels_train_1hot = one_hot(labels_train).cuda()  # 将训练标签转为one-hot编码
            labels_test_1hot = one_hot(labels_test).cuda()  # 将测试标签转为one-hot编码



            cropped_images = [crop_based_on_attention(image, generate_attention_map(base_model, image)) for
                              image in images]
            cropped_images = torch.stack(cropped_images)

            s3 = model(cropped_train, cropped_test, labels_train_1hot, labels_test_1hot)  # 进行模型推理

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()



            s3 = s3.view(q_batch_size * num_test_examples, -1)
            labels_test = labels_test.view(q_batch_size * num_test_examples)

            _, preds = torch.max(s3.detach().cpu(), 1)  # 获取预测结果
            acc = (torch.sum(preds == labels_test.detach().cpu()).float()) / labels_test.size(0)  # 计算准确率
            accs.update(acc.item(), labels_test.size(0))  # 更新准确率计算器

            gt = (preds == labels_test.detach().cpu()).float()
            gt = gt.view(q_batch_size, num_test_examples).numpy()  # 将预测结果转为numpy数组
            acc = np.sum(gt, 1) / num_test_examples  # 计算每个样本的准确率
            acc = np.reshape(acc, (q_batch_size))  # 重新调整准确率的形状
            test_accuracies.append(acc)  # 将准确率添加到列表中

    accuracy = accs.avg  # 计算平均准确率
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    stds = np.std(test_accuracies, 0)  # 计算准确率的标准差
    ci95 = 1.96 * stds / np.sqrt(args.epoch_size)  ### 2000  # 计算95%置信区间
    tm = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))  # 获取当前时间
    print('[{}] Accuracy: {:.2%}, std: {:.2%}'.format(tm, accuracy, ci95))  # 打印准确率和标准差

    return accuracy  # 返回准确率


class PromptEmbedding(nn.Module):
    def __init__(self, num_prompts, embed_dim):
        super().__init__()
        # 初始化可学习的 Prompt Embeddings
        self.prompt_embeddings = nn.Parameter(torch.randn(num_prompts, embed_dim))

    def forward(self, x):
        B = x.size(0)  # 获取 batch size
        # 将 Prompt Tokens 扩展到 batch size 一致
        prompt_tokens = self.prompt_embeddings.unsqueeze(0).expand(B, -1, -1)
        return prompt_tokens  # 返回生成的 Prompt Tokens


def generate_attention_map(model, image):
    """生成 ViT 模型的 Attention Map."""
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        outputs = model(image)  # 前向传播计算输出
        attention_weights = model.blocks[-1].attn.get_attention_map(image)  # 获取最后一层的 Attention Weights
        attention_map = attention_weights.mean(dim=1)  # 对多个注意力头求平均，得到最终 Attention Map
    return attention_map  # 返回 Attention Map


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
    # 创建参数解析器

    # ************************************************************
    # 数据集（通用）
    # ************************************************************
    parser.add_argument('--dataset', type=str, default='CUB_200_2011')  # 数据集名称
    parser.add_argument('--model', default='dino_vits', type=str, help='pretrained_model')  # 模型选择，默认为ResNet
    parser.add_argument('--workers', default=6, type=int,
                        help="number of data loading workers (default: 4)")  # 数据加载器的工作线程数
    parser.add_argument('--height', type=int, default=128, help="height of an image (default: 224)")  # 图像高度
    parser.add_argument('--width', type=int, default=128, help="width of an image (default: 224)")  # 图像宽度

    # ************************************************************
    # 优化选项
    # ************************************************************
    parser.add_argument('--max_epoch', type=int, default=100, help="epoch")  # 训练轮次
    parser.add_argument('--optim', type=str, default='sgd', help="optimization algorithm (see optimizers.py)")  # 优化算法选择
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help="initial learning rate")  # 初始学习率
    parser.add_argument('--weight-decay', default=5e-04, type=float, help="weight decay (default: 5e-04)")  # 权重衰减
    parser.add_argument('--start-epoch', default=0, type=int, help="manual epoch number (useful on restarts)")  # 起始轮数
    parser.add_argument('--train-batch', default=1, type=int, help="train batch size")  # 训练多少批
    parser.add_argument('--test-batch', default=1, type=int, help="test batch size")  # 测试批量大小

    # ************************************************************
    # 架构设置
    # ************************************************************
    parser.add_argument('--num_classes', type=int, default=15)  # 类别数

    # ************************************************************
    # 其他
    # ************************************************************
    parser.add_argument('--save-dir', type=str, default='./result/')  # 结果保存目录
    parser.add_argument('--resume', type=str, default=True, metavar='PATH')  # 恢复模型路径
    parser.add_argument('--gpu-devices', default='0', type=str)  # GPU设备编号

    # ************************************************************
    # 损失设置
    # ************************************************************
    parser.add_argument('--weight_global', type=float, default=1.)  # 全局权重
    parser.add_argument('--weight_local', type=float, default=0.1)  # 局部权重

    # ************************************************************
    # FewShot设置
    # ************************************************************
    parser.add_argument('--nKnovel', type=int, default=5, help='number of novel categories')  # 新类别数   5-way
    parser.add_argument('--nExemplars', type=int, default=1,
                        help='number of training examples per novel category.')  # 每个新类别的训练样本数   1-shot
    parser.add_argument('--train_nTestNovel', type=int, default=1 * 10,
                        help='number of test examples for all the novel category when training')  # 训练阶段，每个类别的查询集有 10 个样本
    parser.add_argument('--train_epoch_size', type=int, default=10,
                        help='number of batches per epoch when training')  # 训练时每轮批次数
    parser.add_argument('--nTestNovel', type=int, default=1 * 5,
                        help='number of test examples for all the novel category')  # 测试阶段，每个类别的查询集有 5 个样本
    parser.add_argument('--epoch_size', type=int, default=50, help='number of batches per epoch')  # 每轮批次数

    parser.add_argument('--phase', default='val', type=str, help='use test or val dataset to early stop')  # 阶段选择
    parser.add_argument('--seed', type=int, default=42)  # 随机种子


    args = parser.parse_args()  # 解析参数


    main(args)  # 调用主函数