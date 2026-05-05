---
title: 卷积家族原理 (Convolutions)
tags: [概念, 卷积, 深度学习, CV, 底层神经元]
created: 2026-05-04
updated: 2026-05-04
sources: 2
status: active
---

# 卷积家族原理 (Convolutions)

## 模块整体说明与架构拆解

卷积（Convolution）是深度学习尤其是计算机视觉（CV）和时间序列处理领域的核心操作。其物理本质是通过一个受限的、可学习的窗口（Filter/Kernel）在输入数据上滑动，进行局部的**互相关（Cross-Correlation）计算**（即逐元素乘法并累加），从而提取出特征。

相比于全连接层（Linear），卷积具备两大数据驱动的归纳偏置（Inductive Bias）：
1. **局部感知（Local Locality）**：物理空间或时间上相近的信号往往高度相关，卷积核强制模型只关注局部区域。
2. **权重共享（Weight Sharing）/平移不变性**：无论特征出现在序列或图像的哪个位置，同一个卷积核扫过去都能被激活，极大地减少了参数量。

### 统一的输出尺寸张量计算法则
无论是处理 1D 时间序列、2D 平面图像，还是 3D 时空体数据，卷积操作在**任意单一空间/时间维度上**的输出尺寸 $L_{out}$ 都严格遵循以下数学下取整公式：
$$ L_{out} = \left\lfloor \frac{L_{in} + 2 \times padding - dilation \times (kernel\_size - 1) - 1}{stride} \right\rfloor + 1 $$

---

## 核心基石一：1D 卷积 (`nn.Conv1d`)

### 1. 架构与适用场景
**适用场景**：时间序列预测、语音/音频特征提取、自然语言处理（NLP，如 TextCNN）。
**核心架构**：1D 卷积核只在一个维度（通常是时间 $T$ 或序列长度 $L$）上滑动。

### 2. 逻辑链输入与输出
- **输入**：`X` 张量，形状为 `[Batch_Size, in_channels, L_in]`。
- **输出**：`Y` 张量，形状为 `[Batch_Size, out_channels, L_out]`。

### 3. 第一性原理与直观理解（气象专家的分析器）
**直观理解**：假设你是一名气象专家，手头有一周 7 天的数据，每天记录了 100 种极其复杂的气象指标（如温度、湿度、风速等）。这份输入数据 $X$ 的形状就是 `[1(Batch), 100(in_channels), 7(L_in)]`。
如果你设置 `kernel_size=2`，意味着你每次严格“看连续的两天”。
如果你设置 `out_channels=10`，相当于你准备了 **10 台不同侧重的“短期趋势提炼机”**。
第一天和第二天，10 台机器把 $100 \times 2 = 200$ 个数据吞进去，吐出 10 个分数；接着机器往后滑动一天（步长 `stride=1`），看第二和第三天，再吐出 10 个分数。滑过这一周，你得到了 $7-(2-1)=6$ 个位置的综合评估。
**物理意义**：你的原始数据被升维提炼成了 `[1, 10, 6]` 的高级特征（如：前两天的温度骤降预示着寒潮）。

![](<images/v2-f497f0f3ba649a47082c115be5fc6dec_r.jpg>)

### 4. 底层神经元与张量运算追踪 (Unfold 与 MatMul)
深度学习框架在底层绝对不是用 `for` 循环去“滑动”计算的，而是依靠**矩阵乘法 (Matrix Multiplication)** 来实现极致加速。
假设输入张量 $X \in \mathbb{R}^{1 \times 4 \times 4}$ ($B=1, C_{in}=4, L_{in}=4$)，卷积核大小 $K=2$，步长 $S=1$，无填充。输出通道 $C_{out}=2$。

1.  **准备权重张量**：卷积核权重 $W \in \mathbb{R}^{2 \times 4 \times 2}$（2个输出通道，每个需要4个输入通道，核长为2）。
2.  **权重拉平 (Flatten)**：将 $W$ 直接展平为 2 行的二维矩阵 $W_{flat} \in \mathbb{R}^{2 \times 8}$。每一行代表一台“提炼机”的 8 个权重参数。
3.  **输入滑窗展开 (Unfold/im2col)**：按照 $K=2$ 在输入序列 $L_{in}=4$ 上去切窗口。一共能切出 3 个窗口。把每个窗口里 $C_{in} \times K = 4 \times 2 = 8$ 个数据全部拉直成一列。把这 3 列拼起来，就得到了一个庞大的数据展开矩阵 $M \in \mathbb{R}^{8 \times 3}$。
4.  **终极矩阵相乘**：$Y_{flat} = W_{flat} \times M + Bias$。
    即 $\mathbb{R}^{2 \times 8} \times \mathbb{R}^{8 \times 3} = \mathbb{R}^{2 \times 3}$。
5.  **恢复形状**：将算出的二维矩阵 `reshape` 回 `[1, 2, 3]` 的 3D 张量形状，1D 卷积计算完成！

### 5. 核心代码详解
```python
import torch
import torch.nn as nn

# 参数量 = out_channels * (in_channels * kernel_size) + bias_terms
#        = 4 * (2 * 5) + 4 = 44 个可学习参数
conv = nn.Conv1d(
    in_channels=2, out_channels=4, kernel_size=5, stride=2, padding=2, bias=True
)

# 输入张量流转: [Batch=8, C_in=2, L_in=100]
x = torch.randn(8, 2, 100)

# 输出尺寸演算: L_out = [(100 + 2*2 - 1*(5-1) - 1) / 2] + 1 = [99/2] + 1 = 49 + 1 = 50
y = conv(x) 
print(y.shape) # 输出结果: [8, 4, 50]
```

---

## 核心基石二：2D 卷积 (`nn.Conv2d`)

### 1. 架构与适用场景
**适用场景**：图像分类、目标检测、语义分割等绝大多数基于平面视觉空间特征提取的任务。
**核心架构**：卷积核在 2D 平面（高度 $H$ 和宽度 $W$）上同时滑动。

### 2. 逻辑链输入与输出
- **输入**：`X` 张量，形状为 `[Batch_Size, in_channels, H_in, W_in]`。
- **输出**：`Y` 张量，形状为 `[Batch_Size, out_channels, H_out, W_out]`。

### 3. 第一性原理与直观理解（多通道特征的坍缩融合）
很多初学者认为 2D 卷积就是拿一个 2D 的矩形矩阵在图像上滑。**这是极其严重的误解**。

**原理解读**：假设我们有一张 $H \times W \times 3$ 的 RGB 图像（输入通道 $C_{in}=3$），我们希望通过 Conv2D 生成 128 个输出特征图（$C_{out}=128$）。
1. 事实上，这里的 128 个 Filter（滤波器）每一个都是 **3D 的小立方体**（例如它的真实形状是 $C_{in} \times K_h \times K_w = 3 \times 3 \times 3$）。
2. 当这个 $3 \times 3 \times 3$ 的立方体在图像的 $H$ 和 $W$ 平面上滑动时，它的三层厚度会**同时**与 RGB 三个通道对应的 $3 \times 3$ 像素进行逐元素相乘，得到 $3 \times 3 \times 3 = 27$ 个乘积结果。
3. **坍缩融合**：这 27 个结果会被直接累加求和（Sum），最后只吐出 **1 个标量**！
4. **结论**：1 个 Filter 无论输入有多少个通道，它扫过全图后只会产生 **1 个单通道**的二维特征图。128 个 Filter 就会产生 128 层的输出。这就是 Conv2D **“在空间上做局部卷积，在通道上做全连接加和”**的本质。

![](<images/v2-0411ccbcb5529b2855478d619ac78d9d_b.gif>)
*(动图：单通道 2D 卷积滑动)*

![](<images/v2-25eeac38abda918fc403bafcf1eadd3b_b.gif>)
*(动图：多通道 2D 卷积在计算完毕后，强行将深度的结果相加坍缩为一个单通道特征图)*

### 4. 核心代码详解
```python
# 参数量 = out_channels * (in_channels * k_h * k_w) + bias
#        = 16 * (3 * 3 * 3) + 16 = 448
conv2d = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=1)

# 输入: [B=8, C_in=3, H=64, W=64]
x = torch.randn(8, 3, 64, 64)

# 经典的 Same Padding: 当 stride=1, pad=k//2 时，宽高经过计算严格不变。
# H_out = [(64 + 2*1 - 1*(3-1) - 1) / 1] + 1 = 64
y = conv2d(x) 
print(y.shape) # 输出: [B=8, C_out=16, H=64, W=64] 
```

---

## 核心基石三：3D 卷积 (`nn.Conv3d`)

### 1. 架构与适用场景
**适用场景**：多模态大模型的视频时空预处理入口（如 Qwen2.5-VL 的时空切块器）、视频动作识别（将时间 T 视为深度 D）、3D 医学图像分割（CT/MRI 有物理 Z 轴层数）。
**核心架构**：卷积核在深度（$D$）、高度（$H$）和宽度（$W$）三个维度上同时滑动。

### 2. 逻辑链输入与输出
- **输入**：`X` 张量，形状为 `[Batch_Size, in_channels, Depth, Height, Width]`。
- **输出**：`Y` 张量，形状为 `[Batch_Size, out_channels, D_{out}, H_{out}, W_{out}]`。

### 3. 第一性原理：3D 卷积与 2D 卷积在时间轴处理上的本质不同
这是理解多模态视频预处理最核心的地方。
假设我们有一段视频，我们把连续的 3 帧画面叠在一起作为输入。输入形状变为 `[Batch, 9(3帧*3RGB), H, W]`。

*   **如果使用 2D 卷积强行处理**：它的滤波器会在 9 个通道上同时相乘并**强制求和坍缩**（见上一节原理）。输出的特征图将完全丢失时间顺序的信息，它只是把 3 帧画面的重影揉碎在了一个静态平面上。
*   **如果使用 3D 卷积**：输入的形状是 `[Batch, 3(RGB通道), 3(帧数Depth), H, W]`。此时滤波器的形状是 `[out_channels, 3(RGB), k_d, k_h, k_w]`。
    此时，滤波器不仅在平面的 H 和 W 上滑动，**它还会在 Depth (时间/深度) 轴上一步步滑动**！
    每一次滑动，它提取的是连续两/三帧的时空连贯变化（初级光流运动）。而且输出的结果依然是一个保留着时间维度 $D_{out}$ 的 3D 张量。这就叫作提纯出了“时空管状特征 (Tubelet)”。

![](<images/v2-acb01897986b8cf1e62604f2bb76b102_b.gif>)
*(动图：3D 卷积不仅在空间移动，还在深度方向移动)*

### 4. 核心代码详解
```python
# Qwen2.5-VL 中的时空切块器就是 3D 卷积
# 参数量 = out_channels * (in_channels * k_d * k_h * k_w) 
#        = 4 * (1 * 3 * 3 * 3) = 108
conv3d = nn.Conv3d(
    in_channels=1, out_channels=4, kernel_size=(3, 3, 3), 
    stride=(1, 2, 2), # 时间维度不下采样，空间维度高宽各缩小一半
    padding=(1, 1, 1), bias=False
)

# 输入构造: [Batch=2, C_in=1, Depth=8, H=64, W=64]
x = torch.randn(2, 1, 8, 64, 64)

# 前向张量流转
y = conv3d(x)

# 验证尺寸变化
# D_out = [(8  + 2*1 - 1*(3-1) - 1) / 1] + 1 = 8
# H_out = [(64 + 2*1 - 1*(3-1) - 1) / 2] + 1 = 32
print(y.shape) # 输出: [2, 4, 8, 32, 32]
```

---

## 关联进阶变体简述
1. **1x1 卷积 (Pointwise)**：没有空间感受野，只对跨通道维度进行极速的 Linear 全连接投影，用于升降维与解耦。
2. **转置卷积 (Transposed)**：用于上采样。步长设计不当会产生极其明显的**棋盘格伪影 (Checkerboard Artifacts)**。
3. **空洞卷积 (Dilated)**：通过在卷积核中填 0 白嫖感受野，但盲目叠加同频率空洞会导致像素独立跳跃（Gridding Effect 灾难）。
4. **深度可分离卷积 (Depthwise Separable)**：切断跨通道融合，先分通道算 2D 卷积，再用 1x1 卷积揉合，乘法算力开销断崖式跌至传统 2D 的 1/9。