> 论文：《Sigmoid Loss for Language Image Pre-Training》
>
> 论文链接：https://arxiv.org/pdf/2303.15343
>
> GitHub：[https://github. com/google-research/big\_vision](https://link.zhihu.com/?target=https%3A//github/)

## 1. 背景

[ CLIP(2021, OpenAI)](https://scnajei2ds6y.feishu.cn/wiki/FKvTwjSHxiVHBokxd5ncXxuanch) 和 [ALIGN](https://proceedings.mlr.press/v139/jia21b.html) 使用大量若监督文本对实现对比学习，标准的做法是使用图像-文本对比目标：拉近嵌入空间中的匹配（正）图像-文本表示，同时远离嵌入空间中无关（负）图像-文本表示。

即，给定一个包含图像-文本对的小批量$$B=\left\{\left(I_1, T_1\right),\left(I_2, T_2\right), \ldots\right\}$$，对比学习的目标是鼓励匹配对 $$\left(I_i, T_i\right)$$ 的嵌入相互对齐，同时将不匹配对 $$\left(I_i, T_{j \neq i}\right)$$ 的嵌入分开。

> **<span style="color: rgb(216,57,49); background-color: inherit">以上有一个noisy and imperfect的假设：对于所有图像 </span>$$i$$<span style="color: rgb(216,57,49); background-color: inherit"> ，与另一图像 </span>$$j$$<span style="color: rgb(216,57,49); background-color: inherit"> 相关联的文本与 </span>$$i$$<span style="color: rgb(216,57,49); background-color: inherit"> 无关，反之亦然。</span>**

但是基于softmax的对比学习存在两个问题：

* [ CLIP(2021, OpenAI)](https://scnajei2ds6y.feishu.cn/wiki/FKvTwjSHxiVHBokxd5ncXxuanch) 的 loss 是 batch-level softmax-based 的，softmax不对称，导致需要分别对图像和文本 normalize the pairwise similarity scores，即：

$$-\frac{1}{2|\mathcal{B}|} \sum_{i=1}^{|\mathcal{B}|}(\overbrace{\log \frac{e^{t \mathbf{x}_i\cdot \mathbf{y}_i} }{\sum_{j=1}^{|\mathcal{B}|} e^{t \mathbf{x}_i\cdot \mathbf{y}_j}}}^{\text {image } \rightarrow \text { text softmax }}+\overbrace{\log \frac{e^{t \mathbf{x}_i\cdot \mathbf{y}_i} }{\sum_{j=1}^{|\mathcal{B}|} e^{t \mathbf{x}_j \cdot \mathbf{y}_i}}}^{\text {text } \rightarrow \text { image softmax }})$$

其中$$\mathbf{x}_i=\frac{f\left(I_i\right)}{\left\|f\left(I_i\right)\right\|_2} $$和$$\mathbf{y}_i=\frac{g\left(T_i\right)}{\left\|g\left(T_i\right)\right\|_2} $$。并且 $$\text { image model 表示为 } f(\cdot) \text { 并且 text model 表示为 } g(\cdot)$$。

* 为了防止softmax计算溢出，需要对每一个值减去当前batch对最大值，因此需要一次预先的遍历。



## 2. SigLIP

### 2.1 计算方法

SigLIP是一种更简单的替代方案，利用Sigmoid代替基于softmax的对比损失，这种方案**无需计算全局归一化因子。**

基于Sigmoid的损失独立地处理每一个图像-文本对，有效地将学习问题转化为在所有对组合的数据集上的标准二分类问题，其中匹配对 $$\left(I_i, T_i\right)$$ 标记为正例，而所有其他对 $$\left(I_i, T_{j \neq i}\right)$$ 标记为负例。其定义如下：

$$-\frac{1}{|\mathcal{B}|} \sum_{i=1}^{|\mathcal{B}|} \sum_{j=1}^{|\mathcal{B}|} \underbrace{\log \frac{1}{1+e^{z_{i j}\left(-t \mathbf{x}_i \cdot \mathbf{y}_j+b\right)}}}_{\mathcal{L}_{i j}}$$

其中 $$z_{i j}$$ 是给定图像和文本输入的标签，如果它们是匹配的则等于1，否则等于-1。

在初始化阶段，来自众多负例的严重不平衡主导了损失，导致初始优化步骤试图修正这种偏差。为了缓解这一点，SigLIP 引入了一个额外的可学习偏置项 $$b$$ ，类似于温度 $$t$$ 。我们将 $$t^{\prime}$$ 和 $$b$$ 分别初始化为 $$\log 10$$和 $$-10$$ 。这确保了训练大致从先验值附近开始，而无需进行大幅度的过度修正。


算法伪代码：

![](<images/SigLIP-截屏2025-03-24 23.58.34.png>)

### 2.2 高效分块(chunked)计算

对比训练通常使用Data Parallel。当数据分布在$$D$$个设备上时计算损失，需要收集所有嵌入来计算每一个pair之间的cosine相似度，这涉及到昂贵的all-gathers操作，更重要的是，需要实例化一个内存密集型的 $$|B| \times|B|$$ 两两相似度矩阵。

**Sigmoid损失特别适合于一种内存高效**，快速且数值稳定的实现方式，这种方式改善了上述两个问题。将每个设备上的批量大小表示为 $$b=\frac{|B|}{D}$$ ，损失可以重新表述为：

![](<images/SigLIP-截屏2025-03-25 00.10.20.png>)

简单来说，首先计算对应于正例对以及$$b-1$$个负例对的损失部分，然后，跨设备置换embedding，以便每个设备从其相邻设备获取负例（求和B）。然后根据这个分块（求和C）计算损失。这是在每个设备上独立完成的，因此每个设备根据其本地批量$$b$$计算损失。然后，可以在所有设备上简单地将损失相加（求和A）。

这个过程的示意图如图，这个过程和[Ring All-reduce](https://zhuanlan.zhihu.com/p/504957661)是类似的：

![](<images/SigLIP-截屏2025-03-25 00.23.28.png>)

**在任何给定时刻的内存成本从 $$|B|^2$$ 降低到了 $$b^2$$，使SigLIP能够在相对较少的设备上实现超过一百万的批量大小进行训练。**



## 3. 实验

与其他公开发布的模型的比较，SigLIP模型在零样本分类和检索任务上均以显著优势超越所有先前的模型，例如OpenCLIP和CLIP。与同期的EVA-CLIP和CLIPA-v2相比，SigLIP-L在低分辨率和高分辨率情况下全面表现出更佳性能

![](<images/SigLIP-截屏2025-03-25 00.39.43.png>)

