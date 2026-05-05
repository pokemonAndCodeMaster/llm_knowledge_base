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

---

## 核心基石：1D, 2D, 3D 卷积的微观解剖

### 1. 1D 卷积 (`nn.Conv1d`)
*   **适用场景**：时间序列预测、语音/音频特征提取、自然语言处理（早期的 TextCNN）。
*   **逻辑链输入与输出**：
    *   **输入**：`(Batch_Size, in_channels, length)`
    *   **输出**：`(Batch_Size, out_channels, L_{out})`
    *   **输出长度公式**：$L_{out} = \left\lfloor \frac{length + 2 \times padding - dilation \times (kernel\_size - 1) - 1}{stride} \right\rfloor + 1$
*   **直观理解（通俗比喻：气象专家）**：
    假设你有一周 7 天的数据，每天记录 100 种气象指标（温度、风速等），形状为 `(100, 7)`。
    `kernel_size=2` 意味着你每次“看两天”；`out_channels=10` 意味着你准备了 10 台不同的“提炼机”。
    第一天和第二天交给 10 台机器，产出 10 个分数；第二天和第三天再产出 10 个分数... 滑过一周，就得到了 6 个两天的趋势分数。数据从“7天×100维”变成了“6个两天窗口×10维”的高层特征。
*   **张量底层的矩阵流转（Unfold 与 MatMul）**：
    假设输入 $X \in \mathbb{R}^{1 \times 4 \times 4}$ ($B=1, C_{in}=4, L_{in}=4$)，卷积核 $K=2$，输出通道 $C_{out}=2$。
    1.  **滑窗展开 (Unfold)**：将长度 4 的序列按 $K=2$ 切出 3 个窗口。将每个窗口的 $4 \times 2 = 8$ 个元素展平。得到矩阵 $M \in \mathbb{R}^{8 \times 3}$。
    2.  **权重拉平**：将卷积核 $W \in \mathbb{R}^{2 \times 4 \times 2}$ 展平为 $W_{flat} \in \mathbb{R}^{2 \times 8}$。
    3.  **矩阵乘法**：$Y_{flat} = W_{flat} \times M + Bias$。得出结果后 Reshape 回 $(1, 2, 3)$。这就是深度学习框架底层实现卷积的极致优化手段。

### 2. 2D 卷积 (`nn.Conv2d`)
*   **适用场景**：图像分类、目标检测、语义分割等平面空间任务。
*   **逻辑链输入与输出**：
    *   **输入**：`(Batch_Size, in_channels, Height, Width)`
    *   **输出**：`(Batch_Size, out_channels, H_{out}, W_{out})`
*   **多通道特征融合的第一性原理**：
    对于一张 $H \times W \times 3$ 的 RGB 图像，假设我们需要 128 个输出通道。
    这 128 个 Filter（滤波器）并不是 2D 的，它们实际上是 **3D** 的（形状为 $3 \times 3 \times 3$）。
    每一次滑动，Filter 里的 3 个切片会分别和 R、G、B 三个通道做逐元素乘法相加，得到 3 个标量，**然后这 3 个标量会被直接相加合并为一个单一的标量**。
    因此，1 个 Filter 扫过全图，无论输入有多少个通道，它只产生 **1 个单通道**的二维特征图。128 个 Filter 就会产生 128 个二维特征图，最终堆叠出输出张量。

### 3. 3D 卷积 (`nn.Conv3d`)
*   **适用场景**：视频动作识别、3D 医学图像分割（CT/MRI）、三维点云。
*   **逻辑链输入与输出**：
    *   **输入**：`(Batch_Size, in_channels, Depth, Height, Width)`（这里的 Depth 可以是视频的帧数 T，也可以是 CT 的层数）。
    *   **输出**：`(Batch_Size, out_channels, D_{out}, H_{out}, W_{out})`
*   **原理与 2D 的本质区别**：
    2D 卷积在处理多通道（比如 3 帧视频叠在一起作为输入通道）时，滑动的维度只有 H 和 W，它会把 3 帧的信息强制在通道维度求和，输出结果**没有深度/时间维度**。
    而 3D 卷积核的形状是 `(k_d, k_h, k_w)`，它不仅在 H 和 W 上滑动，**还在 D (深度/时间) 轴上滑动**。这就意味着它能在保留时间结构的前提下，提取时间连续性特征（即时空 Tubelet 嵌入）。

---

## 进阶变体详解

### 1. 1x1 卷积 (Pointwise Convolution)
*   **模块说明**：看似极其简单的操作（核大小为 1x1），但在 GoogLeNet/Inception 和 ResNet 中起到决定性作用。
*   **物理意义**：它在空间维度（H, W）上没有任何感受野，它的本质是对**通道维度**进行跨通道的线性组合。
*   **三大作用**：降维（减少参数计算量）、升维（低维嵌入）、非线性增加（1x1 卷积后通常接 ReLU）。LeCun 曾说，CNN 中没有全连接层，全连接层本质上就是 $1 \times 1$ 卷积。

### 2. 转置卷积 / 反卷积 (Transposed Convolution)
*   **模块说明**：将小的特征图上采样（Up-sampling）映射为大的特征图。常用于 GAN 生成器或图像分割解码器。注意：它**不是**卷积的数学逆运算，只是在形状和连通性上进行了转置。
*   **致命缺陷与伪影**：如果 $kernel\_size$ 不能被 $stride$ 整除，会导致滑动窗口在某些像素上重叠次数过多，输出图像上会出现明显的**棋盘格伪影 (Checkerboard Artifacts)**。
    ![](<images/v2-02c59b0b621efd10bfab0d726866bfb3_r.jpg>)

### 3. 空洞卷积 / 扩张卷积 (Dilated/Atrous Convolution)
*   **模块说明**：在不增加参数量和计算量的前提下，指数级扩大感受野。常用于语义分割（如 DeepLab）。
*   **实现原理**：在卷积核的元素之间插入“空洞”（即填充 0）。当 $dilation=l$ 时，实际参与计算的感受野扩大为 $l \times (k-1) + 1$。
    ![](<images/v2-b3413dd0b3d01ad9bc188012c3024c45_r.jpg>)
*   **致命缺陷与修复 (Gridding Effect & HDC)**：如果仅仅盲目叠加相同的 dilation rate（如全是 2），会导致卷积操作不连续，像素像棋盘一样互相独立（Gridding Effect）。业界常用 HDC（Hybrid Dilated Convolution）设计锯齿状的膨胀率（如 [1, 2, 5]）来弥补空隙。

### 4. 可分离卷积 (Separable Convolution)
为了极致压缩参数和计算量（如 MobileNet），将标准的 3D 卷积核操作拆解。
*   **深度可分离卷积 (Depthwise Separable Convolution)**：包含两步：
    1.  **Depthwise 卷积**：让 $groups = in\_channels$，即每一个输入通道都由一个专属的 2D 卷积核独立处理。不进行跨通道混合。
    2.  **Pointwise 卷积**：使用普通的 $1 \times 1$ 卷积，在通道维度上将刚才独立的特征混合起来，改变通道数。
*   **算力对比**：假设使用 $3 \times 3$ 卷积核，深度可分离卷积的计算量仅为标准 2D 卷积的约 **1/9**。

### 5. 分组卷积 (Grouped Convolution)
*   **模块说明**：最早出现在 AlexNet 中（为了把模型拆到两块显存极小的 GPU 上训练）。
*   **实现原理**：将输入通道和输出通道平均分成 $G$ 组。第 $i$ 组的滤波器只负责对输入特征的第 $i$ 组进行卷积。参数量下降为 $1/G$。

### 6. 随机分组卷积 (Shuffled Grouped Convolution)
*   **模块说明**：ShuffleNet 提出的架构。解决纯分组卷积的痛点：“不同组之间老死不相往来，信息无法流通”。
*   **实现原理**：在两次分组卷积之间，插入一个 **Channel Shuffle** 操作。将上一层的通道分组后交错重排，强制让下一层的每一组都能“看”到上一层所有组的特征。
    ![](<images/v2-6863240c8c635652f38ef988cf3a8b8a_1440w.jpg>)

---

## 关联概念
- ➡️ 多模态预处理的联合时空建模：[[conv3d_时空切块器]]
- ➡️ ViT 抛弃卷积走向全局注意力：[[vit_核心原理与结构]]