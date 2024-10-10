import torch
import torch.nn as nn
import torch.nn.functional as F



cos = nn.CosineSimilarity(dim=1, eps=1e-6)  # 定义余弦相似度计算器


def Xcos(ftrain, ftest):
    B, n2, n1, C, H, W = ftrain.size()  # 获取输入张量的大小

    ftrain = Long_alignment(ftrain, ftest)  # 长尾对齐

    ftrain = ftrain.view(-1, C, H, W).permute(0, 2, 3, 1)  # 调整形状和维度顺序
    ftest = ftest.view(-1, C, H, W).permute(0, 2, 3, 1)  # 调整形状和维度顺序

    ftrain = ftrain.contiguous().view(-1, ftrain.size(3))  # 重新调整形状
    ftest = ftest.contiguous().view(-1, ftest.size(3))  # 重新调整形状

    cos_map = 10 * cos(ftrain, ftest).view(B * n2, n1, -1)  # 计算余弦相似度并进行调整

    return cos_map  # 返回余弦相似度图


def Long_alignment(support_x, query_x):
    b, q, s, c, h, w = support_x.shape  # 获取张量形状信息
    support_x = F.normalize(support_x, p=2, dim=-3, eps=1e-12)  # 对支持集张量进行L2范数归一化
    query_x = F.normalize(query_x, p=2, dim=-3, eps=1e-12)  # 对查询集张量进行L2范数归一化
    support_x = support_x.view(b, q, s, c, h * w)  # 调整支持集张量形状
    query_x = query_x.view(b, q, s, c, h * w).transpose(3, 4)  # 调整查询集张量形状并转置

    Mt = torch.matmul(query_x, support_x)  # 计算点积得到相似度矩阵

    Mt = F.softmax(Mt, dim=4)  # 对相似度矩阵进行softmax归一化

    support_x = support_x.transpose(3, 4)  # 调整支持集张量形状

    align_support = torch.matmul(Mt, support_x)  # 计算对齐的支持集张量
    align_support = align_support.transpose(3, 4)  # 调整对齐的支持集张量形状

    return align_support  # 返回对齐的支持集张量

