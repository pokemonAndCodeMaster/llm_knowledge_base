---
title: 卷积家族原理 (Convolutions)
tags: [概念, 卷积, 深度学习, CV]
created: 2026-05-04
updated: 2026-05-04
sources: 2
status: active
---

# 卷积家族原理 (Convolutions)

## 模块整体说明

卷积（Convolution）是深度学习尤其是计算机视觉（CV）领域的核心基础操作。其本质是通过一个可学习的滤波器（Filter/Kernel）在输入数据上滑动，进行局部的**互相关（Cross-Correlation）**计算（即逐元素乘法和加法），从而提取出如边缘、纹理等局部特征。

相比于全连接层（Linear/Dense）或全局注意力（Attention），卷积具备两大极其重要的归纳偏置（Inductive Bias）：
1. **局部感知（Local Locality）**：认为相近的像素/信号往往有强相关性。
2. **权重共享（Weight Sharing）/平移不变性**：同一个特征（如猫耳朵）无论出现在图像的哪个位置，都能被同一个滤波器识别。

### 统一的输出尺寸计算法则
无论是处理 1D 时间序列、2D 平面图像，还是 3D 时空体数据，其任意维度上的输出尺寸 $L_{out}$ 都严格遵循以下下取整公式：
$$ L_{out} = \left\lfloor \frac{L_{in} + 2 \times padding - dilation \times (kernel\_size - 1) - 1}{stride} \right\rfloor + 1 $$

---

## 核心变体详解

### 1. 转置卷积 / 反卷积 (Transposed Convolution)
*   **模块说明**：将小的特征图上采样（Up-sampling）映射为大的特征图。常用于 GAN 生成器或图像分割解码器。注意：它**不是**卷积的数学逆运算，只是在形状和连通性上进行了转置。
*   **实现原理**：将正常的卷积操作写成一个巨大的稀疏矩阵乘法 $Y = C X$，那么转置卷积就是用该矩阵的转置去乘以输入 $Y_{up} = C^T X_{small}$。
*   **致命缺陷与伪影**：如果 $kernel\_size$ 不能被 $stride$ 整除，会导致滑动窗口在某些像素上重叠次数过多，输出图像上会出现明显的**棋盘格伪影 (Checkerboard Artifacts)**。
    ![](<images/v2-02c59b0b621efd10bfab0d726866bfb3_r.jpg>)

### 2. 空洞卷积 / 扩张卷积 (Dilated/Atrous Convolution)
*   **模块说明**：在不增加参数量和计算量的前提下，指数级扩大感受野。常用于语义分割（如 DeepLab）。
*   **实现原理**：在卷积核的元素之间插入“空洞”（即填充 0）。当 $dilation=l$ 时，实际参与计算的感受野扩大为 $l \times (k-1) + 1$。
    ![](<images/v2-b3413dd0b3d01ad9bc188012c3024c45_r.jpg>)
*   **致命缺陷与修复 (Gridding Effect & HDC)**：如果仅仅盲目叠加相同的 dilation rate（如全是 2），会导致卷积操作不连续，像素像棋盘一样互相独立（Gridding Effect）。业界常用 HDC（Hybrid Dilated Convolution）设计锯齿状的膨胀率（如 [1, 2, 5]）来弥补空隙。

### 3. 可分离卷积 (Separable Convolution)
为了极致压缩参数和计算量（如 MobileNet），将标准的 3D 卷积核操作拆解。
*   **深度可分离卷积 (Depthwise Separable Convolution)**：包含两步：
    1.  **Depthwise 卷积**：让 $groups = in\_channels$，即每一个输入通道都由一个专属的 2D 卷积核独立处理。不进行跨通道混合。
    2.  **Pointwise 卷积**：使用普通的 $1 \times 1$ 卷积，在通道维度上将刚才独立的特征混合起来，改变通道数。
*   **算力对比**：假设使用 $3 \times 3$ 卷积核，深度可分离卷积的计算量仅为标准 2D 卷积的约 **1/9**。

### 4. 分组卷积 (Grouped Convolution)
*   **模块说明**：最早出现在 AlexNet 中（为了把模型拆到两块显存极小的 GPU 上训练）。现用于 ResNeXt 等架构中以提升参数效率。
*   **实现原理**：将输入通道和输出通道平均分成 $G$ 组。第 $i$ 组的滤波器只负责对输入特征的第 $i$ 组进行卷积。这使得参数量直接下降为原来的 $1/G$。
*   **物理意义**：不同的组往往会自发学习到不同的特征子空间（如 AlexNet 中一组学到了黑白纹理，另一组学到了色彩特征），起到了类似正则化的效果。

### 5. 随机分组卷积 (Shuffled Grouped Convolution)
*   **模块说明**：ShuffleNet 提出的架构。解决纯分组卷积的痛点：“不同组之间老死不相往来，信息无法流通”。
*   **实现原理**：在两次分组卷积之间，插入一个 **Channel Shuffle** 操作。将上一层的通道分组后交错重排，强制让下一层的每一组都能“看”到上一层所有组的特征。
    ![](<images/v2-6863240c8c635652f38ef988cf3a8b8a_1440w.jpg>)

---

## 关联概念
- ➡️ 多模态预处理的联合时空建模：[[conv3d_时空切块器]]
- ➡️ ViT 抛弃卷积走向全局注意力：[[vit_核心原理与结构]]