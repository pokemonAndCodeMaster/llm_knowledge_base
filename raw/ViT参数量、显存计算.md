## 1. ViT参数量计算

以ViT-Base（[google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)）为例：

```python
ViT-Base: 
图片大小为224x224，patch大小为16x16
layers=12,hidden_size=768,MLP_size=3072,heads=12,params=86M
```

* Patch embedding

patch\_dim = 16\*16\*3, dim = hidden\_size = 768，所以参数量为`768*768`

注意这里这个层用conv和linear其实是一样的

* Attention

w\_q,w\_k.w\_v,w\_o

dim\*dim\*4\*layers = `768*768*4*12`

* FFN

dim\*MLP\_size\*2\*layers=`768*3072*2*12`

* LN

dim\*2\*2\*layer=`768*2*2*12`

* MLP\_head: dim\*2=`768*2`

* Postional Embedding

num\_patch\*patch\_dim = `(14*14+1)*768`

注意有一个单独分类的mlp\_head

将以上参数加和，得到总参数量为：85.124352M

> 其他量级模型的参数参考:
>
> $$\begin{array}{|llllll|}
> \hline \text { 模型 } & \text { 层数 } & \text { 隐藏层大小 } D & \text { MLP大小 } & \text { 头数 } & \text { 参数量 } \\
> \hline \text {ViT－Base } & 12 & 768 & 3072 & 12 & 86 \mathrm{M} \\
> \hline \text {ViT－Large } & 24 & 1024 & 4096 & 16 & 307 \mathrm{M} \\
> \hline \text {ViT－Huge } & 32 & 1280 & 5120 & 16 & 632 \mathrm{M} \\
> \hline
> \end{array}$$



## 2. ViT显存计算

这个和LLM是一样的，唯一不同就是推理阶段没有KV Cache这种消耗了。

### 2.1 训练阶段显存分析

> 先说结论，对于参数量为 M (B)的模型：
>
> * **单精度（fp32）训练**：显存消耗约为20M (GB)
>
> * **混合精度（fp32+bf16 or fp32+fp16）训练**：显存消耗理论值为16M (GB)，实际操作中消耗为20M (GB)
>
> * **bf16训练**：显存消耗约为10M (GB)

训练阶段显存分成两个部分，主要可以估计的是Model States和activation：

* Model States

(1) **权重&#x20;**=**&#x20; $$M*dtype\_bytes$$**

(2) **梯度&#x20;**=**&#x20;**$$M*dtype\_bytes$$

(3) **优化器**(对于AdamW) = $$3*M*dtype\_bytes$$ （AdamW中的参数副本、AdamW中的momentum、AdamW中的variances）

* Residual States

(1) **Activation：**

激活值即每一个算子的输出值，用于反向传播计算因此需要缓存。激活值因不同的attention和ffn结构有所不同，但是有一个结论是，激活值大小与 $$layers \times hidden\_dimensions\times  seq\_length \times  batch\_size $$成正比，GPT-2的激活值大约为:$$12\times layers \times hidden\_dimensions\times  seq\_length \times  batch\_size $$

当然大家也可以自己动手计算一下具体的表达式，比如 llama-13b 的激活值表达式为 $$l(34bsd+5bhs)$$，其中 $$bhs$$远小于前面一项，故可以省略。

> 现在深度训练中普遍使用checkpoint技术节约显存，即前向计算时，只保留部分算子的激活值。反向更新时，需要其他算子的激活值时，再重新对其进行前向计算，得到其激活值。分为两种方式：
>
> * full checkpoint：对所有算子都进行Activation checkpointing，等于走了两遍完整的前向计算，虽然将内存消耗减小到平方根的量级，即从60GB－＞8GB；但是会带来 $$36 \%$$ 的重新计算开销。
>
> * Selective checkpointing：只对那些计算时间小，占显存多的op（如attention部分的op）进行 Activation checkpointing。重新计算开销从 $$36\%$$－＞$$4\%$$
>
> 在transformers中使用这一特性十分简单，参考后面代码。

(2) **Temporary buffers：**&#x7528;于梯度融合（Gradient Fusion）、All-Reduce（跨设备梯度同步）、梯度范数计算（Gradient Norm Computation）

(3) **Memory Fragmentation：**&#x7C7B;似于OS内存管理中的外部碎片问题。



#### 2.1.1 单精度训练

全局$$dtype\_bytes = 4$$，带入得总显存开销 20M (GB)

#### 2.1.2 混合精度训练

混合精度训练（MPT）可以将Model States 中的 (1) 和 (2) 的 $$dtype\_bytes$$ 变为 2，即使用float16或者bfloat16，优化器 $$dtype\_bytes$$ 保持为 4。从而将显存降低到 16M (GB)

但是参考[transformers文档](https://developer.aliyun.com/article/1563212)和accelerate文档，这个说法是不对的：**混合精度训练**是一种旨在通过利用较低精度数值格式来处理某些变量来优化训练模型的**计算效率**的技术。注意看，这里说的是计算效率，意味着其实在这两个框架内显存是不会节约的，甚至在bs小的情况下显存还会多出一些消耗：

> [原因解释](https://discuss.huggingface.co/t/why-does-setting-fp16-true-not-save-memory-as-expected/22400)：Where did you read it would save you memory? Training with mixed precision will be faster, but does not save memory when you train large models, because instead of having 1 model in FP32 in GPU RAM, you get 1 copy in FP32 and 1 copy in FP16 (so 1.5 times the memory).&#x20;

所以说，其实在启用混合精度训练的时候，还需要保存一份float32的模型副本，导致显存开销实质上是 20M (GB)，可以后面的参考实践代码动手试试。



#### 2.1.3 **bf16训练**

全局 $$dtype\_bytes = 2$$，带入得总显存开销 10M (GB)

参考[文档](https://hub.baai.ac.cn/view/38665)，transformers 通过如下 `bf16=True` 的设置， `bfloat16` 也可以被用在优化器上。



### 2.2 推理阶段显存分析

> 先说结论，对于参数量为 M (B)的模型：
>
> * **半精度（fp16 or bf16）推理**：显存消耗约为2M (GB)
>
> * **单精度（fp32）推理**：显存消耗约为4M (GB)

推理阶段的显存主要分为Model States

* Model States

这里只涉及模型权重，为 $$M*dtype\_bytes$$

