import torch
from skimage import measure
import torch.nn.functional as F
import math
import torch.nn as nn

# def AOLM(feature_maps):
#     width = feature_maps.size(-1)
#     height = feature_maps.size(-2)
#     A = torch.sum(feature_maps, dim=1, keepdim=True)
#     print(A.shape)
#
#     a = torch.mean(A, dim=[2, 3], keepdim=True)
#     M = (A > a).float()
#
#
#     coordinates = []
#     for i, m in enumerate(M):
#         mask_np = m.cpu().numpy().reshape(height, width)
#         component_labels = measure.label(mask_np)
#
#         properties = measure.regionprops(component_labels)
#         areas = []
#         for prop in properties:
#             areas.append(prop.area)
#         if len(areas)==0:
#             bbox = [0,0,height, width]
#         else:
#
#             max_idx = areas.index(max(areas))
#
#             bbox = properties[max_idx].bbox
#
#         temp = 224/width              #############84改成224
#         temp = math.floor(temp)
#         x_lefttop = bbox[0] * temp - 1
#         y_lefttop = bbox[1] * temp - 1
#         x_rightlow = bbox[2] * temp- 1
#         y_rightlow = bbox[3] * temp - 1
#         if x_lefttop < 0:
#             x_lefttop = 0
#         if y_lefttop < 0:
#             y_lefttop = 0
#
#         coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
#         coordinates.append(coordinate)
#     return coordinates

#新的修改
import torch
import math
import numpy as np
from skimage import measure

import torch
import math
import numpy as np
from skimage import measure


def AOLM(feature_maps):
    width = feature_maps.size(-1)
    height = feature_maps.size(-2)
    A = torch.sum(feature_maps, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float()


    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(height, width)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        if len(areas)==0:
            bbox = [0,0,height, width]
        else:

            max_idx = areas.index(max(areas))

            bbox = properties[max_idx].bbox

        temp = 84/width
        temp = math.floor(temp)
        x_lefttop = bbox[0] * temp - 1
        y_lefttop = bbox[1] * temp - 1
        x_rightlow = bbox[2] * temp- 1
        y_rightlow = bbox[3] * temp - 1
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0

        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates

    return coordinates


def crop_featuremaps(raw_imgs, feature_maps):
    batch_size = feature_maps.size(0)  # 获取批量大小（特征图的第一个维度）
    coordinates = AOLM(feature_maps)  # 使用AOLM函数获取特征图中的坐标
    crop_imgs = torch.zeros([batch_size, 3, 224, 224]).cuda()  # 创建用于存储裁剪图像的张量，并将其移动到GPU      ###########################84改成224
    for i in range(batch_size):  # 遍历每个批次
        [x0, y0, x1, y1] = coordinates[i]  # 获取第i个样本的坐标
        crop_imgs[i:i + 1] = F.interpolate(
            raw_imgs[i:i + 1, :, x0:(x1 + 1), y0:(y1 + 1)],  # 从原始图像中裁剪对应的区域
            size=(224, 224),  # 将裁剪后的图像大小调整为84x84                ##################84改成224
            mode='bilinear',  # 使用双线性插值进行缩放
            align_corners=True  # 对齐角点
        )

    return crop_imgs

def drop_featuremaps(feature_maps):
    width = feature_maps.size(-1)  # 获取特征图的宽度
    height = feature_maps.size(-2)  # 获取特征图的高度
    A = torch.sum(feature_maps, dim=1, keepdim=True)  # 对特征图在通道维度上求和，保持维度不变
    a = torch.max(A, dim=3, keepdim=True)[0]  # 在宽度维度上找到最大值，并保持维度不变
    a = torch.max(a, dim=2, keepdim=True)[0]  # 在高度维度上找到最大值，并保持维度不变
    threshold = 0.85  # 定义阈值
    M = (A <= threshold * a).float()  # 根据阈值生成掩码，将A中不大于阈值*a的元素置为1，其余置为0
    fm_temp = feature_maps * M  # 使用掩码M来屏蔽特征图中的某些区域
    return fm_temp  # 返回处理后的特征图

