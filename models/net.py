import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models.resnet12 import resnet12
from models.conv4 import ConvNet4
from .xcos import Xcos
from .BAS import crop_featuremaps, drop_featuremaps
from models.CBAM import CBAMBlock
from src.models.vit_models import ViT
from src.models.vit_prompt.vit import PromptedTransformer
from models.crop_img import Crop

class Trainer():
    def __init__(self):
        super().__init__()

        self.is_training = True
        # self.nFeat = self.base.nFeat  # 特征数为base的特征数

        # self.clasifier1 = nn.Linear(self.nFeat, num_classes)  # 第一个分类器，线性层
        # self.clasifier2 = nn.Linear(self.nFeat, num_classes)  # 第二个分类器，线性层

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层

    
    def duenei(self, model, xtrain, ytrain, xtest, ytest, args):  # 前向传播函数定义，接收训练集和测试集数据以及它们对应的标签
        batch_size, num_train = xtrain.size(0), xtrain.size(1)  # 获取训练集批次大小和样本数量
        num_test = xtest.size(1)  # 获取测试集样本数量
        K = ytrain.size(2)  # 获取标签类别数量
        ytrain = ytrain.transpose(1, 2)  # 调整标签张量的维度顺序

        cropped = Crop()

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))  # 重新调整训练集数据张量的形状
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))  # 重新调整测试集数据张量的形状
        images = torch.cat((xtrain, xtest), 0)  # 将训练集和测试集数据合并
        attentions = model(images, 1)
        crops = cropped.crop_img(args, images, attentions)
        crops = crops.cuda()

        # crop_tests = cropped.crop_img(args, xtest, attentions)

        # if use_gpu:
        #     cropped_xtrain = cropped_image_train.cuda()
        #     cropped_xtest = cropped_image_test.cuda()

        # cropped_images = torch.cat((cropped_xtrain, cropped_xtest), 0)  # 将训练集和测试集数据合并

        # f = model(images)
        # crop_f = model(cropped_images)

        if self.is_training:  # 如果是训练模式
            # flatten_f = self.avgpool(f)  # 对丢弃后的特征图进行平均池化操作
            # flatten_f = flatten_f.view(flatten_f.size(0), -1)  # 将平均池化后的特征展平
            # flatten_crop_f = self.avgpool(crop_f)  # 对裁剪后的特征图进行平均池化操作
            # flatten_crop_f = flatten_crop_f.view(flatten_crop_f.size(0), -1)  # 将平均池化后的特征展平
            glo1 = model(images)  # 全局特征分类器1
            glo2 = model(crops)  # 全局特征分类器2

        ftrain = model(xtrain, 0, 1)


        ftest = model(xtest, 0, 1)

        # ftrain = f[:batch_size * num_train]  # 获取训练集特征
        # ftrain = ftrain.view(batch_size, num_train, -1)  # 调整训练集特征张量的形状
        #
        # # 获取原型
        # ftrain = torch.bmm(ytrain, ftrain)  # 使用标签与训练集特征进行矩阵乘法，以获取原型向量
        # ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))  # 对原型向量进行归一化处理
        # ftrain = ftrain.view(batch_size, -1, *f.size()[1:])  # 调整原型向量张量的形状
        #
        # ftest = f[batch_size * num_train:]  # 获取测试集特征
        # ftest = ftest.view(batch_size, num_test, *f.size()[1:])  # 调整测试集特征张量的形状
        #
        # f1 = ftrain.unsqueeze(1).repeat(1, num_test, 1, 1, 1, 1)  # 将原型向量扩展为与测试集特征相同的形状
        # f2 = ftest.unsqueeze(2).repeat(1, 1, K, 1, 1, 1)  # 将测试集特征扩展为与原型向量相同的形状
        #
        # ftrain_crop = crop_f[:batch_size * num_train]  # 获取裁剪后的训练集特征
        # ftrain_crop = ftrain_crop.view(batch_size, num_train, -1)  # 调整裁剪后的训练集特征张量的形状

        ftrain_crop = crops[:batch_size * num_train]
        ftrain_crop = ftrain_crop.view(batch_size, num_train, -1)  # 调整裁剪后的训练集特征张量的形状
        # ftest_crop = crops[batch_size:]
        # ftest_crop = ftest_crop.view(batch_size, num_train, -1)  # 调整裁剪后的训练集特征张量的形状

        # 获取原型
        ftrain_crop = torch.bmm(ytrain, ftrain_crop)  # 使用标签与裁剪后的训练集特征进行矩阵乘法，以获取裁剪后的原型向量
        ftrain_crop = ftrain_crop.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain_crop))  # 对裁剪后的原型向量进行归一化处理
        ftrain_crop = ftrain_crop.view(batch_size, -1, *f.size()[1:])  # 调整裁剪后的原型向量张量的形状

        # ftest_crop = crop_f[batch_size * num_train:]  # 获取裁剪后的测试集特征
        ftest_crop = crops[batch_size * num_train:]
        ftest_crop = ftest_crop.view(batch_size, num_test, *f.size()[1:])  # 调整裁剪后的测试集特征张量的形状

        f1_crop = ftrain_crop.unsqueeze(1).repeat(1, num_test, 1, 1, 1, 1)  # 将裁剪后的原型向量扩展为与裁剪后的测试集特征相同的形状
        f2_crop = ftest_crop.unsqueeze(2).repeat(1, 1, K, 1, 1, 1)  # 将裁剪后的测试集特征扩展为与裁剪后的原型向量相同的形状

        similar2 = Xcos(f1, f2)  # 计算特征相似度
        similar1 = Xcos(f1_crop, f2_crop)  # 计算裁剪后的特征相似度

        s1 = similar1.view(-1, K, h * w)  # 调整裁剪后的特征相似度张量的形状
        s2 = similar2.view(-1, K, h * w)  # 调整特征相似度张量的形状

        if not self.training:  # 如果不是训练模式
            return s1.sum(-1) * 0.5 + s2.sum(-1) * 0.5  # 返回特征相似度的加权和

        # return crop_f  # 返回特征相似度、全局特征1和全局特征2
        return s1, s2, glo1, glo2


