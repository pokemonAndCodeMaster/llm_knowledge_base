# 让研究人员绞尽脑汁的Transformer位置编码
**作者:** 苏剑林 (科学空间)
**日期:** 2021-02-03
**原文链接:** https://kexue.fm/archives/8130

不同于RNN、CNN等模型，对于Transformer模型来说，位置编码的加入是必不可少的，因为纯粹的Attention模块是无法捕捉输入顺序的，即无法区分不同位置的Token。为此我们大体有两个选择：
1、想办法将位置信息融入到输入中，这构成了绝对位置编码的一般做法；
2、想办法微调一下Attention结构，使得它有能力分辨不同位置的Token，这构成了相对位置编码的一般做法。

## 绝对位置编码
形式上来看，绝对位置编码是相对简单的一种方案。一般来说，绝对位置编码会加到输入中：在输入的第 $k$ 个向量 $x_k$ 中加入位置向量 $p_k$ 变为 $x_k+p_k$，其中 $p_k$ 只依赖于位置编号 $k$。

### 1. 训练式 (Trainable)
直接将位置编码当作可训练参数。现在的BERT、GPT等模型所用的就是这种位置编码。缺点是没有外推性（预测长度不能超过训练长度）。

### 2. 三角函数式 (Sinusoidal)
Google 的《Attention is All You Need》提出的显式解：
$$ \begin{cases} p_{k,2i} = \sin(k/10000^{2i/d}) \ p_{k,2i+1} = \cos(k/10000^{2i/d}) \end{cases} $$
由于 $\sin(\alpha+\beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$ 等性质，位置 $\alpha+\beta$ 的向量可以表示成位置 $\alpha$ 和位置 $\beta$ 的向量组合。

### 3. 递归式 (FLOATER)
用微分方程 (ODE) 的方式建模位置编码。具有较好的外推性，但牺牲了一定的并行性。

## 相对位置编码
相对位置编码通常是在算 Attention 的时候考虑当前位置与被 Attention 的位置的相对距离。

### 1. 经典式 (Shaw et al.)
Google 的论文《Self-Attention with Relative Position Representations》提出。将 $R_{i,j}^K$ 改为只依赖于相对距离 $i-j$，并且通常会进行截断。

### 2. XLNET式 (Transformer-XL)
直接将绝对位置向量替换为相对位置向量 $R_{i-j}$，并将查询位置信息替换为两个可训练的全局向量 $u, v$。

### 3. T5式
仅仅在 Attention 矩阵的基础上加一个可训练的偏置项 $\beta_{i,j}$：
$$ 	ext{Attention}(Q, K) = \frac{QK^	op}{\sqrt{d}} + \beta_{i,j} $$
T5 对相对位置进行了“分桶”处理（邻近精细，远距离粗糙）。

### 4. DeBERTa式
改进点在于扔掉了“位置-位置”交互项，保留了“内容-位置”和“位置-内容”项。DeBERTa 在 SuperGLUE 榜单上表现优异。

## 其他位置编码
- **CNN式**：位置信息可能是由于 Zero Padding 泄漏的。
- **复数式**：将词向量映射到复平面，利用复数乘法的旋转特性注入位置信息。

## 总结
绝对位置编码形式简单但外推性存疑；相对位置编码灵活性大，但实现相对复杂。苏剑林随后提出了旋转位置编码 (RoPE)，它通过绝对位置的操作达到了相对位置的效果。
