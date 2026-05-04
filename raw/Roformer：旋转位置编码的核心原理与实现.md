# Transformer升级之路：2、高效、外推性好的位置编码
**作者:** 苏剑林 (科学空间)
**日期:** 2021-03-23
**原文链接:** https://kexue.fm/archives/8823 (注：原 ID 8823 关联 8265)

在上一篇文章中，我们对原始的 Sinusoidal 位置编码做了推导。本文将详细介绍我们自研的 **Roformer** 模型，它的核心是旋转位置编码 (Rotary Positional Embedding, **RoPE**)。

## 核心思想：旋转变换
RoPE 的核心思想是：**通过绝对位置的操作，达到相对位置的效果**。

我们希望寻找一个函数 $f(q, m)$，使得：
$$ \langle f(q, m), f(k, n) \rangle = g(q, k, m-n) $$

### 1. 二维推导
对于二维向量 $[x, y]$，我们将其视为复数 $x+iy$。将其乘以旋转因子 $e^{im\theta}$：
$$ f(q, m) = q e^{im\theta} = (q_0 \cos m\theta - q_1 \sin m\theta) + i(q_0 \sin m\theta + q_1 \cos m\theta) $$
内积结果为：
$$ \langle q e^{im\theta}, k e^{in\theta} \rangle = \text{Re}[q e^{im\theta} (k e^{in\theta})^*] = \text{Re}[q k^* e^{i(m-n)\theta}] $$
内积只取决于相对位置 $m-n$！

### 2. 多维扩展
对于 $d$ 维向量，我们将维度两两分组，每一组施加不同的旋转频率 $\theta_i = 10000^{-2i/d}$。
旋转矩阵 $R_m$ 为一个分块对角矩阵：
$$ R_m = \text{diag} \left( \begin{pmatrix} \cos m\theta_0 & -\sin m\theta_0 \\ \sin m\theta_0 & \cos m\theta_0 \end{pmatrix}, \dots, \begin{pmatrix} \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\ \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1} \end{pmatrix} \right) $$

## RoPE 的优势
1. **外推性**：RoPE 具有天然的外推能力。
2. **远程衰减**：随着相对距离变大，内积结果趋向于衰减。实验证明，RoPE 能有效捕捉邻近词的依赖。
3. **实现简单**：只需要对 $q$ 和 $k$ 进行一次线性旋转变换。

## 线性 Attention 兼容性
RoPE 是目前唯一一种可以无缝集成到线性 Attention 中的相对位置编码，因为它不操作 Attention 矩阵，而是直接作用于 $q, k$ 向量。

## 总结
RoPE 结合了绝对位置编码的实现效率和相对位置编码的表达能力。它是目前所有主流大语言模型（Llama, ChatGLM, Qwen）的核心组件。
