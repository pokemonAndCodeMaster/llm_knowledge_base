# 从熵不变性看Attention的Scale操作
**作者:** 苏剑林 (科学空间)
**日期:** 2021-12-21
**原文链接:** https://kexue.fm/archives/8823

当前Transformer架构用的最多的注意力机制，全称为“Scaled Dot-Product Attention”，其中“Scaled”是因为在$Q,K$转置相乘之后还要除以一个$\sqrt{d}$再做Softmax（下面均不失一般性地假设$Q,K,V\in\mathbb{R}^{n	imes d}$）：

\begin{equation}Attention(Q,K,V) = softmax\left(\frac{QK^{	op}}{\sqrt{d}}ight)V\end{equation}

在《浅谈Transformer的初始化、参数化与标准化》中，我们已经初步解释了除以$\sqrt{d}$的缘由。而在这篇文章中，笔者将从“熵不变性”的角度来理解这个缩放操作，并且得到一个新的缩放因子。在MLM的实验显示，新的缩放因子具有更好的长度外推性能。

## 熵不变性
我们将一般的Scaled Dot-Product Attention改写成
\begin{equation}\boldsymbol{o}_i = \sum_{j=1}^n a_{i,j}\boldsymbol{v}_j,\quad a_{i,j}=\frac{e^{ \lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j}}{\sum\limits_{j=1}^n e^{ \lambda \boldsymbol{q}_i\cdot \boldsymbol{k}_j}}\end{equation}
其中$\lambda$是缩放因子。本文提出一个观点：
> 为了使得模型结果能够更好地泛化到未知长度，Attention机制的设计应该使得$a_{i,j}$尽量具备熵不变性。

具体来说，$a_{i,j}$的熵为
\begin{equation}\mathcal{H}_i = -\sum_{j=1}^n a_{i,j}\log a_{i,j}\end{equation}
熵不变性是指，$\mathcal{H}_i$应该对长度$n$不敏感。我们希望熵不变，是希望引入新的token后，已有的token依旧能同样地聚焦到原来的token上，而不希望新token的引入过多地“分摊”了原有的注意力。

## 新的因子
根据熵不变性，可以得到一种新的Scaled Dot-Product Attention：
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{\kappa \log n}{d}QK^{	op}ight)V\end{equation}
为了去掉超参数$\kappa$，假设当$n=512$时退化为普通的$\frac{1}{\sqrt{d}}$，整理得到：
\begin{equation}Attention(Q,K,V) = softmax\left(\frac{\log_{512} n}{\sqrt{d}}QK^{	op}ight)V\end{equation}

### 实验结果
| | n=64 | n=128 | n=256 | n=512 | n=1024 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Attention-O | 43.27 | 36.53 | 23.02 | 15.12 | 11.54 |
| Attention-E | 43.11 | 41.17 | 34.04 | 20.15 | 13.58 |

实验结果显示，Attention-E 在长序列外推上的准确率明显拉开了差距。

## 推导过程
（此处省略部分复杂的统计学推导，核心结论如下）
通过拉普拉斯近似得到：
\begin{equation}\mathcal{H}_i \approx \log n - 0.24\lambda d + \mathcal{O}(1)\end{equation}
因此，为了抵消长度$n$的影响，让$\log n - 0.24\lambda d = 0$，从而得出 $\lambda \propto \frac{\log n}{d}$。

## 文章总结
本文从熵不变性的角度重新推导了Scale操作，得到了一个新的缩放因子。初步实验显示其对长度外推具有更好的结果。
