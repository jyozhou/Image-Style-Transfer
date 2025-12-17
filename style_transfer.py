"""
神经风格迁移（Neural Style Transfer）实现
使用预训练的VGG16网络提取特征，通过优化生成图像使其同时匹配内容图的结构和风格图的纹理
"""
import argparse
from typing import Dict, List, Tuple

import jittor as jt
from jittor import nn, optim
import numpy as np
from PIL import Image
from jittor.models import vgg16


def load_image(path: str, max_size: int = 512) -> jt.Var:
    """
    加载图像并转换为张量格式
    
    功能：
    - 读取图像并转换为RGB格式
    - 如果图像尺寸超过max_size，按比例缩放
    - 将像素值归一化到[0,1]范围
    - 转换为Jittor张量，形状为(1, 3, H, W)，符合深度学习模型输入格式
    """
    img = Image.open(path).convert("RGB")
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
        img = img.resize(new_size, Image.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0  # (H,W,3)
    arr = jt.array(arr).permute(2, 0, 1).unsqueeze(0)
    return arr


def to_pil(t: jt.Var) -> Image.Image:
    """
    将张量转换回PIL图像格式
    
    实现方式：
    - 将张量值限制在[0,1]范围内
    - 从(1,3,H,W)转换为(H,W,3)格式
    - 将浮点值[0,1]映射回整数[0,255]
    - 转换为PIL Image对象用于保存
    """
    t = jt.clamp(t, 0.0, 1.0)
    arr = t[0].numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = (arr * 255.0).clip(0, 255).astype("uint8")
    return Image.fromarray(arr)


def gram_matrix(feat: jt.Var) -> jt.Var:
    """
    计算特征图的Gram矩阵，用于捕获风格信息
    
    作用：
    - Gram矩阵衡量特征通道之间的相关性，能够捕获纹理和风格模式
    - 这是风格迁移的核心：通过匹配Gram矩阵来传递风格特征
    
    实现方式：
    - 将特征图(b,c,h,w)重塑为(b,c,h*w)
    - 计算特征矩阵与其转置的乘积得到Gram矩阵
    - 归一化处理，使结果不受特征图尺寸影响
    """
    b, c, h, w = feat.shape
    x = feat.reshape((b, c, h * w))
    g = jt.bmm(x, x.transpose(0, 2, 1))
    return g / (c * h * w)


class VGGFeatures(nn.Module):
    """
    VGG特征提取器：从预训练的VGG16网络中提取中间层特征
    
    作用：
    - 使用预训练的VGG16作为特征提取器（不更新权重）
    - 提取指定层的特征用于内容损失和风格损失计算
    - 低层特征捕获细节和纹理（风格），高层特征捕获结构（内容）
    """

    def __init__(self, content_layer: str, style_layers: List[str]):
        super().__init__()
        vgg_model = vgg16(pretrained=True)
        self.blocks = vgg_model.features
        self.content_layer = content_layer  # 用于内容匹配的层（通常选择conv4_2）
        self.style_layers = set(style_layers)  # 用于风格匹配的多层（conv1_1到conv5_1）
        # 卷积层名称到VGG features中索引的映射
        self.layer_map = {
            "conv1_1": 0,
            "conv2_1": 5,
            "conv3_1": 10,
            "conv4_1": 17,
            "conv4_2": 19,
            "conv5_1": 24,
        }
        # 冻结VGG参数，只用于特征提取，不参与训练
        for p in self.parameters():
            p.stop_grad()

    def execute(self, x: jt.Var) -> Tuple[Dict[str, jt.Var], Dict[str, jt.Var]]:
        """
        前向传播：提取内容和风格特征
        
        返回：
        - content_feats: 内容特征字典（用于内容损失）
        - style_feats: 风格特征字典（用于风格损失）
        """
        content_feats = {}
        style_feats = {}
        out = x
        for idx, layer in enumerate(self.blocks):
            out = layer(out)
            # 在指定层提取特征
            for name, lid in self.layer_map.items():
                if idx == lid:
                    if name == self.content_layer:
                        content_feats[name] = out
                    if name in self.style_layers:
                        style_feats[name] = out
        return content_feats, style_feats


def style_transfer(
    content_path: str,
    style_path: str,
    output_path: str,
    iters: int = 300,
    content_weight: float = 1e5,
    style_weight: float = 1e7,
    noise_strength: float = 0.1,
    lr: float = 0.02,
) -> None:
    """
    神经风格迁移主函数
    
    实现原理：
    1. 使用预训练VGG16提取内容和风格特征
    2. 初始化生成图像（内容图+噪声，增加风格探索空间）
    3. 通过优化生成图像，使其特征同时匹配：
       - 内容损失：生成图与内容图在conv4_2层的特征差异
       - 风格损失：生成图与风格图在多层Gram矩阵的差异
    4. 使用Adam优化器迭代更新生成图像
    
    参数说明：
    - content_weight: 内容损失权重（alpha），控制内容保留程度
    - style_weight: 风格损失权重（beta），控制风格迁移强度
    - noise_strength: 初始化噪声强度，影响风格探索范围
    """
    if jt.has_cuda:
        jt.flags.use_cuda = 1

    # 加载内容和风格图像
    content_img = load_image(content_path)
    style_img = load_image(style_path, max_size=max(content_img.shape[2], content_img.shape[3]))

    # 初始化生成图像：内容图 + 白噪声
    # 添加噪声有助于跳出局部最优，增强风格迁移效果
    noise = jt.randn_like(content_img) * noise_strength
    generated = jt.clamp(content_img + noise, 0.0, 1.0)
    generated.requires_grad = True

    # 构建VGG特征提取器并提取目标特征
    model = VGGFeatures(content_layer="conv4_2", style_layers=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"])
    style_targets = model(style_img)[1]  # 提取风格图的各层特征
    style_grams = {k: gram_matrix(v) for k, v in style_targets.items()}  # 计算风格图的Gram矩阵
    content_target = model(content_img)[0]["conv4_2"]  # 提取内容图的目标层特征

    # 分层风格权重：低层（颜色/纹理）权重更高，高层权重较低
    # 这样设计是因为低层特征更能捕获颜色和细节纹理，对风格迁移更重要
    style_layer_weights = {
        "conv1_1": 1.0,   # 最高权重，捕获颜色和细节纹理
        "conv2_1": 0.8,   # 高权重
        "conv3_1": 0.5,   # 中等权重
        "conv4_1": 0.3,   # 较低权重
        "conv5_1": 0.1,   # 最低权重
    }

    optimizer = optim.Adam([generated], lr=lr)

    # 迭代优化生成图像
    for i in range(1, iters + 1):
        # 提取生成图的特征
        content_feats, style_feats = model(generated)
        
        # 计算内容损失：生成图与内容图在conv4_2层的MSE损失
        c_loss = nn.mse_loss(content_feats["conv4_2"], content_target)
        
        # 计算风格损失：各层Gram矩阵的加权MSE损失
        s_loss = 0
        for k, feat in style_feats.items():
            g = gram_matrix(feat)
            layer_weight = style_layer_weights.get(k, 1.0)
            s_loss += layer_weight * nn.mse_loss(g, style_grams[k])
        
        # 总损失 = 内容损失 * 内容权重 + 风格损失 * 风格权重
        loss = content_weight * c_loss + style_weight * s_loss

        # 反向传播并更新生成图像
        optimizer.step(loss)

        # 将像素值限制在[0,1]范围内
        with jt.no_grad():
            generated.update(jt.clamp(generated, 0.0, 1.0))
        
        # 关键修复：clamp 操作后重新设置 requires_grad，否则后续迭代无法计算梯度
        generated.requires_grad = True

        # 定期打印损失信息
        if i % 50 == 0:
            print(f"iter {i}/{iters}  loss={loss.item():.4f}  c={c_loss.item():.4f}  s={s_loss.item():.4f}")

    # 保存生成的风格迁移图像
    to_pil(generated).save(output_path)
    print(f"Saved stylized image to: {output_path}")


def main():
    """
    命令行入口函数
    
    解析命令行参数并调用风格迁移函数
    支持自定义迭代次数、损失权重、噪声强度等超参数
    """
    parser = argparse.ArgumentParser(description="Minimal Jittor Neural Style Transfer")
    parser.add_argument("--content", required=True, help="内容图路径")
    parser.add_argument("--style", required=True, help="风格图路径")
    parser.add_argument("--output", default="output.jpg", help="输出图路径")
    parser.add_argument("--iters", type=int, default=300, help="迭代次数")
    parser.add_argument("--cw", type=float, default=1e5, help="内容损失权重 alpha")
    parser.add_argument("--sw", type=float, default=1e7, help="风格损失权重 beta")
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="生成图初始化噪声强度（0 为纯内容，>0 增加白噪声）",
    )
    parser.add_argument("--lr", type=float, default=0.02, help="Adam 学习率")
    args = parser.parse_args()

    style_transfer(
        content_path=args.content,
        style_path=args.style,
        output_path=args.output,
        iters=args.iters,
        content_weight=args.cw,
        style_weight=args.sw,
        noise_strength=args.noise,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()

"""

python style_transfer.py --content sourceimage/image1.jpg --style sourceimage/style1.jpg --output output1.jpg --iters 2000 --cw 2000 --sw 5000000000 --noise 0.2 --lr 0.01

python style_transfer.py --content sourceimage/image2.jpg --style sourceimage/style2.jpg --output output2.jpg --iters 4000 --cw 5000 --sw 3000000000 --noise 0.15 --lr 0.003

python style_transfer.py --content sourceimage/image3.jpg --style sourceimage/style3.jpg --output output3.jpg --iters 1500 --cw 10000 --sw 1800000000 --noise 0.2 --lr 0.004

python style_transfer.py --content sourceimage/image4.jpg --style sourceimage/style4.jpg --output output4.jpg --iters 3000 --cw 2000 --sw 8000000000 --noise 0.45 --lr 0.01

"""