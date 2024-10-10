import torch
import torch.nn as nn
from utils import transforms as T
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
from PIL import Image
class Crop():
    def __init__(self):
        super().__init__()
        self.preprocessor = self._get_preprocessor()

    def _get_preprocessor(self):
        resize = T.Compose([
            T.Resize((224, 224), interpolation=3),  # 将图像大小调整为 (96, 96)
            T.ToTensor(),  # 将图像转换为张量
        ])
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 定义标准化操作

        def _preprocess(args, image):
            resized = resize(image)  # 调整图像大小
            image = normalize(resized)  # 标准化图像

            w = image.shape[1] - image.shape[1] % args.patch_size  # 调整图像宽度使其能够被patch size整除
            h = image.shape[2] - image.shape[2] % args.patch_size  # 调整图像高度使其能够被patch size整除

            image = image[:, :w, :h].unsqueeze(0)  # 调整图像尺寸并增加batch维度
            resized = resized[:, :w, :h].permute(1, 2, 0)  # 调整已调整大小的图像

            w_featmap = image.shape[-2] // args.patch_size  # 计算特征图的宽度
            h_featmap = image.shape[-1] // args.patch_size  # 计算特征图的高度

            return ((image, resized), (w_featmap, h_featmap))  # 返回处理后的图像及其特征图大小

        return _preprocess  # 返回预处理器函数

    def crop_img(self, args, images, attentions):

        to_pil = ToPILImage()
        # 遍历批次中的每张图像并转换为 PIL 图像
        pil_images = [to_pil(images[i]) for i in range(images.shape[0])]
        attentions = attentions[-1]  # 获取最后一层的自注意力图
        crops = []  # 初始化 crops
        # print(attentions)  # [15, 12, 197, 197]
        for i, image in enumerate(pil_images):
            # 处理图像
            # image.show()  # 显示图像
            with torch.no_grad():  # 禁用梯度计算
                (image_tensor, resized_image), (w_featmap, h_featmap) = self.preprocessor(args, image)  # 使用预处理器处理图像
                # print(image_tensor.shape, resized_image.shape, w_featmap, h_featmap)
                # attentions = model.get_last_selfattention(image_tensor)  # 获取最后一层的自注意力图
                last_attentions = attentions[i:i + 1]  # 注意 [i:i+1] 使形状变为 [1, 12, 197, 197]
            # print(attentions)
            nh = last_attentions.shape[1]  # 获取注意力图中的头数

            # 处理注意力图
            print(last_attentions.shape)
            last_attentions = last_attentions[0, :, 0, 1:].reshape(nh, -1)  # 只保留每个头的[CLS]到其他token的注意力值
            # print(last_attentions.shape)
            if args.threshold != 0:  # 如果设置了阈值
                val, idx = torch.sort(last_attentions)  # 对注意力值进行排序
                val /= torch.sum(val, dim=1, keepdim=True)  # 归一化
                cumval = torch.cumsum(val, dim=1)  # 计算累积和
                th_attn = cumval > (1 - args.threshold)  # 选择超过阈值的注意力
                idx2 = torch.argsort(idx)  # 根据索引进行排序
                for head in range(nh):  # 对每个注意力头处理
                    th_attn[head] = th_attn[head][idx2[head]]  # 根据阈值调整注意力
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()  # 调整形状
                th_attn = \
                nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[
                    0]  # 使用插值放大到原图大小
                last_attentions = th_attn.sum(0)  # 合并所有头的注意力
            else:
                last_attentions = last_attentions.reshape(nh, w_featmap, h_featmap)  # 调整形状
                last_attentions = \
                nn.functional.interpolate(last_attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[
                    0]  # 使用插值放大
                last_attentions = last_attentions.sum(0)  # 合并所有头的注意力

            # 裁剪图像
            crop_transform = T.CenterCrop((args.output_height, args.output_width))  # 定义中心裁剪
            h, w, _ = resized_image.size()  # 获取图像的高宽
            conv_weight = torch.ones((1, 1, args.sum_span, args.sum_span), dtype=torch.float32)  # 定义卷积核
            pad_size = args.sum_span // 2  # 计算填充大小
            padded_attention = nn.functional.pad(last_attentions, (pad_size, pad_size, pad_size, pad_size),
                                                 value=0)  # 对注意力图进行填充
            padded_attention = padded_attention.cuda()
            conv_weight = conv_weight.cuda()
            scores = nn.functional.conv2d(padded_attention.unsqueeze(0).unsqueeze(0), conv_weight)[0, 0]  # 计算卷积得分

            max_index = (scores == torch.max(scores)).nonzero()[0]  # 获取得分最高的位置
            max_h_start = h - args.output_height  # 计算裁剪起始点的最大值（高度）
            max_w_start = w - args.output_width  # 计算裁剪起始点的最大值（宽度）

            # 根据得分计算裁剪起始位置
            h_start = min(max(max_index[0] + (args.sum_span // 2) - (args.output_height // 2), 0), max_h_start)
            w_start = min(max(max_index[1] + (args.sum_span // 2) - (args.output_width // 2), 0), max_w_start)

            # 根据得分裁剪图像
            score_cropped = resized_image[h_start:h_start + args.output_height, w_start:w_start + args.output_width, :]

            score_cropped = score_cropped.permute(2, 0, 1)

            score_cropped = score_cropped.unsqueeze(0)  # 转换为 4D Tensor (1, 3, H, W)
            score_cropped_resized = F.interpolate(score_cropped, size=(224, 224), mode="bilinear", align_corners=True)  # 调整大小

            crops.append(score_cropped_resized)  # 将调整大小后的裁剪图像存入列表

        crops = torch.cat(crops, dim=0)  # 拼接成形状 [15, 3, 224, 224]
        return crops

