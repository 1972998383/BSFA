import os  # 导入os模块，用于文件夹遍历
import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms  # 导入Pytorch的图像变换工具
from absl import logging, app, flags  # 导入absl库用于日志记录、应用程序管理和标志定义
from PIL import Image  # 导入Pillow库，用于处理图像
import matplotlib.pyplot as plt  # 导入matplotlib库，用于绘图

# 定义命令行参数
FLAGS = flags.FLAGS
flags.DEFINE_string("input_folder", "CUB_200_2011", help="Folder containing images to crop")  # 定义输入文件夹路径
flags.DEFINE_string("output_folder", "cropped_output4", help="Folder to save cropped images")  # 定义输出文件夹路径
flags.DEFINE_integer("image_size", 480, help="input image size")  # 定义输入图片的尺寸

flags.DEFINE_string("model", "dino_vits", help="model name")  # 定义模型名称
flags.DEFINE_integer("patch_size", 16, help="dino patch size")  # 定义patch大小

flags.DEFINE_float("threshold", 0.4, help="threshold (set 0 to disable)")  # 定义注意力图的阈值
flags.DEFINE_integer("sum_span", 30, "sum span")  # 定义卷积核的大小，用于计算得分
flags.DEFINE_integer("output_width", 480, "output image width")  # 定义输出图片的宽度
flags.DEFINE_integer("output_height", 360, "output image height")  # 定义输出图片的高度


def main(argv):
    if FLAGS.input_folder == "":  # 检查输入文件夹参数是否为空
        raise ValueError("You should pass --input_folder=FOLDER_PATH argument")  # 如果为空则抛出异常

    model_name = f"{FLAGS.model}{FLAGS.patch_size}"  # 组合模型名称和patch大小
    logging.info(f"model: {model_name}")  # 记录模型信息
    logging.info(f"patch size: {FLAGS.patch_size}")  # 记录patch大小
    logging.info(f"image size: ({FLAGS.image_size})")  # 记录图像尺寸

    logging.info("Load dino model")  # 记录加载模型信息
    model = torch.hub.load("facebookresearch/dino:main", model_name)  # 从Torch Hub加载DINO模型
    preprocessor = _get_preprocessor()  # 获取预处理函数

    # 遍历输入文件夹中的所有文件
    for root, dirs, files in os.walk(FLAGS.input_folder):  # 使用os.walk遍历文件夹
        for file in files:  # 遍历每个文件
            if file.endswith((".jpg", ".png", ".jpeg")):  # 检查是否为支持的图片格式
                img_path = os.path.join(root, file)  # 构造图片的完整路径
                logging.info(f"Processing image: {img_path}")  # 记录处理的图片路径

                # 加载图像
                with open(img_path, "rb") as f:  # 以二进制模式打开图片
                    img = Image.open(f)  # 使用PIL打开图像
                    img = img.convert("RGB")  # 将图像转换为RGB模式

                # 处理图像
                with torch.no_grad():  # 禁用梯度计算
                    (img_tensor, resized_img), (w_featmap, h_featmap) = preprocessor(img)  # 使用预处理器处理图像
                    # print(img_tensor.shape, resized_img.shape, w_featmap, h_featmap)
                    attentions = model.get_last_selfattention(img_tensor)  # 获取最后一层的自注意力图
                nh = attentions.shape[1]  # 获取注意力图中的头数

                # 处理注意力图
                print(attentions.shape)
                attentions = attentions[0, :, 0, 1:].reshape(nh, -1)  # 只保留每个头的[CLS]到其他token的注意力值
                print(attentions.shape)
                if FLAGS.threshold != 0:  # 如果设置了阈值
                    val, idx = torch.sort(attentions)  # 对注意力值进行排序
                    val /= torch.sum(val, dim=1, keepdim=True)  # 归一化
                    cumval = torch.cumsum(val, dim=1)  # 计算累积和
                    th_attn = cumval > (1 - FLAGS.threshold)  # 选择超过阈值的注意力
                    idx2 = torch.argsort(idx)  # 根据索引进行排序
                    for head in range(nh):  # 对每个注意力头处理
                        th_attn[head] = th_attn[head][idx2[head]]  # 根据阈值调整注意力
                    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()  # 调整形状
                    th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=FLAGS.patch_size, mode="nearest")[0]  # 使用插值放大到原图大小
                    attentions = th_attn.sum(0)  # 合并所有头的注意力
                else:
                    attentions = attentions.reshape(nh, w_featmap, h_featmap)  # 调整形状
                    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=FLAGS.patch_size, mode="nearest")[0]  # 使用插值放大
                    attentions = attentions.sum(0)  # 合并所有头的注意力

                # 裁剪图像
                crop_transform = pth_transforms.CenterCrop((FLAGS.output_height, FLAGS.output_width))  # 定义中心裁剪
                h, w, _ = resized_img.size()  # 获取图像的高宽
                conv_weight = torch.ones((1, 1, FLAGS.sum_span, FLAGS.sum_span), dtype=torch.float32)  # 定义卷积核
                pad_size = FLAGS.sum_span // 2  # 计算填充大小
                padded_attention = nn.functional.pad(attentions, (pad_size, pad_size, pad_size, pad_size), value=0)  # 对注意力图进行填充
                scores = nn.functional.conv2d(padded_attention.unsqueeze(0).unsqueeze(0), conv_weight)[0, 0]  # 计算卷积得分

                max_index = (scores == torch.max(scores)).nonzero()[0]  # 获取得分最高的位置
                max_h_start = h - FLAGS.output_height  # 计算裁剪起始点的最大值（高度）
                max_w_start = w - FLAGS.output_width  # 计算裁剪起始点的最大值（宽度）

                # 根据得分计算裁剪起始位置
                h_start = min(max(max_index[0] + (FLAGS.sum_span // 2) - (FLAGS.output_height // 2), 0), max_h_start)
                w_start = min(max(max_index[1] + (FLAGS.sum_span // 2) - (FLAGS.output_width // 2), 0), max_w_start)

                # 根据得分裁剪图像
                score_cropped = resized_img[h_start:h_start + FLAGS.output_height, w_start:w_start + FLAGS.output_width, :]

                center_cropped = crop_transform(resized_img.permute(2, 0, 1)).permute(1, 2, 0)  # 中心裁剪

                # 保存裁剪后的图像
                output_subfolder = root.replace(FLAGS.input_folder, FLAGS.output_folder)  # 创建对应的输出子文件夹
                if not os.path.exists(output_subfolder):  # 如果输出文件夹不存在
                    os.makedirs(output_subfolder)  # 创建输出文件夹
                output_image_path = os.path.join(output_subfolder, file)  # 构建输出图像路径

                logging.info(f"Saving cropped image to: {output_image_path}")  # 记录保存路径
                _plot_and_save(resized_img, attentions, scores, center_cropped, score_cropped, output_image_path)  # 保存图像


def _get_preprocessor():
    resize = pth_transforms.Compose([
        pth_transforms.Resize(FLAGS.image_size),  # 调整图像大小
        pth_transforms.ToTensor(),  # 转换为Tensor
    ])
    normalize = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 定义标准化操作

    def _preprocess(img):
        resized = resize(img)  # 调整图像大小
        img = normalize(resized)  # 标准化图像

        w = img.shape[1] - img.shape[1] % FLAGS.patch_size  # 调整图像宽度使其能够被patch size整除
        h = img.shape[2] - img.shape[2] % FLAGS.patch_size  # 调整图像高度使其能够被patch size整除

        img = img[:, :w, :h].unsqueeze(0)  # 调整图像尺寸并增加batch维度
        resized = resized[:, :w, :h].permute(1, 2, 0)  # 调整已调整大小的图像
        w_featmap = img.shape[-2] // FLAGS.patch_size  # 计算特征图的宽度
        h_featmap = img.shape[-1] // FLAGS.patch_size  # 计算特征图的高度

        return ((img, resized), (w_featmap, h_featmap))  # 返回处理后的图像及其特征图大小

    return _preprocess  # 返回预处理器函数


def _plot_and_save(img, attention, scores, center_cropped, score_cropped, output_image_path):
    fig = plt.figure(figsize=[25, 10])  # 创建画布

    ax = fig.add_subplot(1, 5, 1)  # 添加子图1
    ax.imshow(img)  # 显示原始图像
    ax.set_title("Original Image")  # 设置标题

    ax = fig.add_subplot(1, 5, 2)  # 添加子图2
    ax.imshow(attention)  # 显示注意力图
    ax.set_title("Attention")  # 设置标题

    ax = fig.add_subplot(1, 5, 3)  # 添加子图3
    ax.imshow(scores)  # 显示得分图
    ax.set_title("Scores for Cropping")  # 设置标题

    ax = fig.add_subplot(1, 5, 4)  # 添加子图4
    ax.imshow(center_cropped)  # 显示中心裁剪图
    ax.set_title("Center Cropped")  # 设置标题

    ax = fig.add_subplot(1, 5, 5)  # 添加子图5
    ax.imshow(score_cropped)  # 显示基于得分裁剪的图
    ax.set_title("Cropped using Attention")  # 设置标题

    fig.savefig(output_image_path, facecolor='white', transparent=False)  # 保存图片


if __name__ == "__main__":
    app.run(main)  # 运行主程序
