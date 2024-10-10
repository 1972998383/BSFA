from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import os.path as osp


class FGSCR(object):
    """
    Dataset statistics:
    # 64 * 600 (train) + 16 * 600 (val) + 20 * 600 (test)
    """

    # dataset_dir = '/home/10701006/Datasets/Fine_grained/CUB_200_2011'
    dataset_dir = 'FGSCR'
    # dataset_dir = 'D:\Datasets\Fine_grained\CUB_200_2011'
    def __init__(self):
        super(FGSCR, self).__init__()
        # 设置训练、验证、测试数据集文件夹路径
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.val_dir = os.path.join(self.dataset_dir, 'val')
        self.test_dir = os.path.join(self.dataset_dir, 'test')

        # 处理训练、验证、测试数据集文件夹，获取数据集、标签映射和标签列表
        self.train, self.train_labels2inds, self.train_labelIds = self._process_dir(self.train_dir)
        self.val, self.val_labels2inds, self.val_labelIds = self._process_dir(self.val_dir)
        self.test, self.test_labels2inds, self.test_labelIds = self._process_dir(self.test_dir)
        # print(self.train_labelIds)
        # 计算数据集统计信息
        self.num_train_cats = len(self.train_labelIds)
        num_total_cats = len(self.train_labelIds) + len(self.val_labelIds) + len(self.test_labelIds)
        num_total_imgs = len(self.train + self.val + self.test)

        print("=> FGSCR loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # cats | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(len(self.train_labelIds), len(self.train)))
        print("  val      | {:5d} | {:8d}".format(len(self.val_labelIds), len(self.val)))
        print("  test     | {:5d} | {:8d}".format(len(self.test_labelIds), len(self.test)))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_cats, num_total_imgs))
        print("  ------------------------------")

    def _check_before_run(self):
        """在深入之前检查所有文件是否可用"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.val_dir):
            raise RuntimeError("'{}' is not available".format(self.val_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path):  # 处理目录的函数

        cat_container = sorted(os.listdir(dir_path))  # 对目录中的内容进行排序
        cats2label = {cat: label for label, cat in enumerate(cat_container)}  # 将类别映射到标签

        dataset = []  # 数据集列表
        labels = []  # 标签列表
        for cat in cat_container:  # 遍历类别容器
            for img_path in sorted(os.listdir(os.path.join(dir_path, cat))):  # 遍历类别下的图像路径
                if '.bmp' not in img_path:  # 如果不是jpg格式的文件
                    continue
                label = cats2label[cat]  # 获取类别对应的标签
                dataset.append((os.path.join(dir_path, cat, img_path), label))  # 添加图像路径和标签到数据集
                labels.append(label)  # 添加标签到标签列表

        labels2inds = {}  # 标签到索引的映射
        for idx, label in enumerate(labels):  # 遍历标签列表
            if label not in labels2inds:  # 如果标签不在映射中
                labels2inds[label] = []  # 初始化映射
            labels2inds[label].append(idx)  # 将索引添加到对应标签的列表中

        labelIds = sorted(labels2inds.keys())  # 对标签映射中的键进行排序
        return dataset, labels2inds, labelIds   ### 图片路径+label  每个类有哪些图片（id）  有哪些label（类）


