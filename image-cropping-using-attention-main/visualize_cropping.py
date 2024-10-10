import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
from absl import logging, app, flags
from PIL import Image
import matplotlib.pyplot as plt

# 定义FLAGS，允许从命令行传递参数
FLAGS = flags.FLAGS
flags.DEFINE_string("image", "sample_images/6.jpg", help="Image path to check cropping")  # 定义图片路径参数
flags.DEFINE_integer("image_size", 480, help="input image size")  # 定义输入图片大小

flags.DEFINE_string("model", "dino_vits", help="model name")  # 定义模型名称参数
flags.DEFINE_integer("patch_size", 16, help="dino patch size")  # 定义模型的patch size参数

flags.DEFINE_float("threshold", 0, help="threhsold (set 0 to disable)")  # 定义阈值，0为禁用
flags.DEFINE_string("output", "output.png", help="Image output path")  # 定义输出图片路径
flags.DEFINE_integer("sum_span", 30, "sum span")  # 定义卷积的sum span，用于计算得分
flags.DEFINE_integer("output_width", 480, "output image size")  # 定义输出图片的宽度
flags.DEFINE_integer("output_height", 360, "output image size")  # 定义输出图片的高度


def main(argv):
    if FLAGS.image == "":
        raise ValueError("You should pass --image=IMAGE_PATH argument")  # 如果没有传递图片路径，抛出异常

    model_name = f"{FLAGS.model}{FLAGS.patch_size}"  # 组合模型名称
    logging.info(f"model: {model_name}")  # 记录模型信息
    logging.info(f"patch size: {FLAGS.patch_size}")  # 记录patch size信息
    logging.info(f"image size: ({FLAGS.image_size})")  # 记录图片大小信息

    logging.info("Load dino model")
    model = torch.hub.load("facebookresearch/dino:main", model_name)  # 从Torch Hub加载DINO模型
    preprocessor = _get_preprocessor()  # 获取预处理器

    logging.info("Load image")
    with open(FLAGS.image, "rb") as f:  # 打开并加载图像
        img = Image.open(f)
        img = img.convert("RGB")  # 转换为RGB模式

    logging.info("forward image")
    with torch.no_grad():  # 禁用梯度计算
        (img, resized_img), (w_featmap, h_featmap) = preprocessor(img)  # 预处理图像，获取特征图尺寸
        attentions = model.get_last_selfattention(img)  # 获取最后一层自注意力输出
    nh = attentions.shape[1]  # 获取头的数量（多头注意力）

    logging.info("modify attention for plot")
    # 只保留输出补丁的注意力
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)  # 只保留每个头的[CLS] token到其他token的注意力

    if FLAGS.threshold != 0:
        # 保留一定比例的注意力质量
        val, idx = torch.sort(attentions)  # 对注意力值进行排序
        val /= torch.sum(val, dim=1, keepdim=True)  # 归一化每个头的注意力值
        cumval = torch.cumsum(val, dim=1)  # 计算累积和
        th_attn = cumval > (1 - FLAGS.threshold)  # 只保留超过阈值的注意力
        idx2 = torch.argsort(idx)  # 按照排序后的索引重新排列
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]  # 根据阈值选择注意力
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()  # 调整形状为特征图大小
        # 插值到原始大小
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=FLAGS.patch_size, mode="nearest")[0]
        attentions = th_attn.sum(0)  # 将多头注意力叠加
    else:
        attentions = attentions.reshape(nh, w_featmap, h_featmap)  # 调整形状为特征图大小
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=FLAGS.patch_size, mode="nearest")[0]
        attentions = attentions.sum(0)  # 将多头注意力叠加

    logging.info("Crop")
    crop_transform = pth_transforms.CenterCrop((FLAGS.output_height, FLAGS.output_width))  # 定义中心裁剪变换
    h, w, _ = resized_img.size()  # 获取图片的高度和宽度

    conv_weight = torch.ones((1, 1, FLAGS.sum_span, FLAGS.sum_span), dtype=torch.float32)  # 创建卷积核
    pad_size = FLAGS.sum_span // 2  # 计算填充大小
    padded_attention = nn.functional.pad(attentions, (pad_size, pad_size, pad_size, pad_size), value=0)  # 填充注意力图
    scores = nn.functional.conv2d(padded_attention.unsqueeze(0).unsqueeze(0), conv_weight)  # 计算卷积得分
    scores = scores[0,0]  # 移除额外的维度

    max_index = (scores==torch.max(scores)).nonzero()[0]  # 获取得分最大的位置
    logging.info(f"Center point: {max_index}")  # 记录中心点位置

    max_h_start = h - FLAGS.output_height  # 计算裁剪高度的最大起始点
    max_w_start = w - FLAGS.output_width  # 计算裁剪宽度的最大起始点

    h_start = min(max(max_index[0] + (FLAGS.sum_span // 2) - (FLAGS.output_height // 2), 0), max_h_start)  # 确定高度的裁剪起点
    w_start = min(max(max_index[1] + (FLAGS.sum_span // 2) - (FLAGS.output_width // 2), 0), max_w_start)  # 确定宽度的裁剪起点

    score_cropped = resized_img[h_start:h_start+FLAGS.output_height, w_start:w_start+FLAGS.output_width,:]  # 根据得分裁剪图片
    center_cropped = crop_transform(resized_img.permute(2, 0, 1)).permute(1, 2, 0)  # 执行中心裁剪

    logging.info("Save plot")
    _plot_and_save(resized_img, attentions, scores, center_cropped, score_cropped)  # 保存图片和结果


def _get_preprocessor():
    resize = pth_transforms.Compose([
        pth_transforms.Resize(FLAGS.image_size),  # 调整图片大小
        pth_transforms.ToTensor(),  # 转换为Tensor
    ])
    normalize = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 定义标准化操作

    def _preprocess(img):
        resized = resize(img)  # 调整图片大小
        img = normalize(resized)  # 标准化图片

        # 使图像尺寸可以被patch size整除
        w = img.shape[1] - img.shape[1] % FLAGS.patch_size  # 调整宽度
        h = img.shape[2] - img.shape[2] % FLAGS.patch_size  # 调整高度

        img = img[:, :w, :h].unsqueeze(0)  # 调整图像尺寸并增加批次维度
        resized = resized[:, :w, :h].permute(1, 2, 0)  # 调整已调整大小的图像

        w_featmap = img.shape[-2] // FLAGS.patch_size  # 计算特征图的宽度
        h_featmap = img.shape[-1] // FLAGS.patch_size  # 计算特征图的高度

        return ((img, resized), (w_featmap, h_featmap))  # 返回处理后的图像和特征图尺寸

    return _preprocess  # 返回预处理函数


def _plot_and_save(img, attention, scores, center_cropped, score_cropped):
    fig = plt.figure(figsize=[25, 10])  # 创建绘图

    ax = fig.add_subplot(1, 5, 1)  # 创建子图1
    ax.imshow(img)  # 显示原始图片
    ax.set_title("original Image")  # 设置标题

    ax = fig.add_subplot(1, 5, 2)  # 创建子图2
    ax.imshow(attention)  # 显示注意力图
    ax.set_title("attention")  # 设置标题

    ax = fig.add_subplot(1, 5, 3)  # 创建子图3
    ax.imshow(scores)  # 显示得分图
    ax.set_title("scores for cropping")  # 设置标题

    ax = fig.add_subplot(1, 5, 4)  # 创建子图4
    ax.imshow(center_cropped)  # 显示中心裁剪结果
    ax.set_title("center cropped")  # 设置标题

    ax = fig.add_subplot(1, 5, 5)  # 创建子图5
    ax.imshow(score_cropped)  # 显示基于注意力裁剪的结果
    ax.set_title("cropped using attention")  # 设置标题

    fig.savefig(f"{FLAGS.output}", facecolor='white', transparent=False)  # 保存结果图像


if __name__ == "__main__":
    app.run(main)  # 启动程序
