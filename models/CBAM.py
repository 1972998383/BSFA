import torch.nn as nn
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):  # 初始化通道注意力模块
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化层
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层
        self.se = nn.Sequential(  # 注意力机制的堆叠卷积层序列
            nn.Conv2d(channel, channel // reduction, 1, bias=False),  # 通道压缩卷积层
            nn.ReLU(),  # ReLU激活函数
            nn.Conv2d(channel // reduction, channel, 1, bias=False)  # 通道恢复卷积层
        )
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):  # 前向传播函数
        max_result = self.maxpool(x)  # 最大池化结果
        avg_result = self.avgpool(x)  # 平均池化结果
        max_out = self.se(max_result)  # 最大池化结果经过注意力机制处理
        avg_out = self.se(avg_result)  # 平均池化结果经过注意力机制处理
        output = self.sigmoid(max_out + avg_out)  # 最大池化和平均池化结果的加权融合
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):  # 初始化空间注意力模块
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)  # 卷积层用于融合最大池化和平均池化结果
        self.sigmoid = nn.Sigmoid()  # Sigmoid激活函数

    def forward(self, x):  # 前向传播函数
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # 在通道维度上取最大值
        avg_result = torch.mean(x, dim=1, keepdim=True)  # 在通道维度上取平均值
        result = torch.cat([max_result, avg_result], 1)  # 拼接最大池化和平均池化结果
        output = self.conv(result)  # 通过卷积层融合结果
        output = self.sigmoid(output)  # Sigmoid激活函数处理融合结果
        return output


class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=49):  # 初始化CBAM模块
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)  # 通道注意力模块
        self.sa = SpatialAttention(kernel_size=kernel_size)  # 空间注意力模块

    def init_weights(self):  # 参数初始化函数
        for m in self.modules():  # 遍历模块
            if isinstance(m, nn.Conv2d):  # 如果是卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化权重
                if m.bias is not None:  # 如果存在偏置
                    init.constant_(m.bias, 0)  # 初始化偏置为0
            elif isinstance(m, nn.BatchNorm2d):  # 如果是批归一化层
                init.constant_(m.weight, 1)  # 初始化批归一化参数
                init.constant_(m.bias, 0)  # 初始化批归一化参数
            elif isinstance(m, nn.Linear):  # 如果是全连接层
                init.normal_(m.weight, std=0.001)  # 使用正态分布初始化权重
                if m.bias is not None:  # 如果存在偏置
                    init.constant_(m.bias, 0)  # 初始化偏置为0

    def forward(self, x):  # 前向传播函数
        b, c, _, _ = x.size()  # 获取输入张量的大小
        residual = x  # 残差连接
        out = x * self.ca(x)  # 经过通道注意力机制
        out = out * self.sa(out)  # 经过空间注意力机制
        return out + residual  # 加上残差连接的结果
