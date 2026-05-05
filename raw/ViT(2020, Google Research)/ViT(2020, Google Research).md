> ViT：多模态任务发展的基础(2020, Google Research)
>
> [Vision Transfromer](https://arxiv.org/abs/2010.11929)

## 1. ViT 概述



> 一句话概括：主要贡献在于将现有架构（Transformers）应用于 CV 领域。**训练方法**和**用于预训练网络的数据集是&#x20;**&#x56;iT 在 ImageNet 上取得与 SOTA 相比优异结果的关键。**论证了 ViT 可以代替CNN，论证了大规模+大模型在图像领域的涌现能力，也为后续两年多模态任务的发展奠定了基石。**

> ViT 的设计遵循最小改动原则，直接将原始 Transformer 应用于图像分类任务。研究者提出，无需总是依赖 CNN，仅凭 Transformer 也能在分类任务中取得优异表现，尤其是在大规模数据集上训练时。经过大规模预训练的 ViT 模型还能在迁移至中等或小规模数据集的任务后，展现出超越 CNN 的性能。

![](<images/ViT(2020, Google Research)-image.png>)



### 1.1 ViT背景与意义



1. **Transformer 在 NLP 领域的成功启发了在 CV 领域的应用**

Transformer在 NLP 领域凭借自注意力机制取得巨大成功，促使研究者探索其在 CV 中的应用。

* **ViT 证明了 Transformer 可以独立于 CNN 处理图像**

ViT 是Google在2020年提出的直接将Transformer应用在图像分类的模型<span style="color: rgb(143,149,158); background-color: inherit">，</span>没有对 Transformer 结果做任何改动，直接将图像划分为若干固定大小的 Patch（图块）作为 token ，同时引入位置编码保留空间信息，输入 Transformer中，通过堆叠的 Transformer 来捕获图像中的局部和全局关系。ViT 摆脱了卷积操作，完全依赖自注意力机制来捕捉图像中的长距离依赖关系，证明了纯 Transformer 也能高效处理图像数据。

* **ViT 大力出奇迹**

&#x20;ViT 在超大规模数据集（如 Google JFT-300M）上进行预训练后，并迁移到中等规模或小规模的数据集（如 ImageNet、CIFAR-100、VTAB 等）时，其性能可超越最先进的 CNN 模型。例如，ViT 在 ImageNet-1K 数据集上取得了 88.55% 的分类准确率，显现出 Transformer 在计算机视觉领域的强大潜力。



### 1.2 ViT 与 CNN 的对比

|            | **ViT**      | **CNN**                |
| ---------- | ------------ | ---------------------- |
| **核心操作**   | 自注意力(全局关联)   | 卷积（局部滤波）               |
| **归纳偏置**   | 无显式偏置，数据驱动   | 强空间局部性、平移等变性           |
| **数据需求**   | 依赖大规模预训练     | 中小规模数据高效               |
| **长程依赖建模** | 单层即可全局交互     | 需深层堆叠或特殊模块(如Non-local) |
| **计算复杂度**  | 平方增长(需分块优化)  | 线性增长(与图像大小)            |
| **可解释性**   | 原生注意力热力图     | 依赖类激活图(CAM)            |
| **典型应用场景** | 大数据预训练、跨模态任务 | 实时检测、边缘设备              |

> **特征建模机制：局部感知 vs 全局关联**
>
> * CNN通过**局部卷积层**逐步扩大感受野，依赖层次堆叠捕获特征，但长程依赖建模受限。
>
> * ViT利用**自注意力机制单层动态建立图像块间的全局关联，即可计算任意位置间的权重，尤其擅长捕捉跨区域的语义联系（如猫的头部与尾巴的协同识别）**，无需深层堆叠。

> **输入处理范式：像素阵列 vs 序列化块**
>
> * CNN直接处理**原始像素矩阵**，依赖卷积核的空间不变性，平等对待所有像素。
>
> * ViT将图像**切割为固定尺寸块（如16×16）并线性映射为序列**，引入可学习位置编码保留空间信息，通过注意力权重动态聚焦关键区域（如分类任务增强主体权重）。

> **数据效率与扩展性：小数据优势 vs 大数据潜能**
>
> * CNN凭借**局部性先验**在小数据场景（如ImageNet-1K）收敛更快且不易过拟合。
>
> * ViT缺乏空间偏置，需**超大规模预训练**（如JFT-300M）才能发挥潜力，但随数据量增长性能显著提升（如ViT-H/14达88.55%准确率），且迁移学习泛化更强，说明其表征更具通用性。

> **计算效率与部署：实时性 vs 可解释性**
>
> * CNN因**线性计算复杂度**和硬件优化，在实时推理（如移动端）占优。
>
> * ViT的自注意力导致**平方复杂度**，高分辨率图像处理开销大，但可通过分块或稀疏注意力优化。
>
> * ViT的**注意力热力图**提供直观解释，CNN则依赖后处理可视化（如CAM）。



> **结构特性与信息流**
>
> * **信息保留**：ViT的**无降采样设计**（如ViT-Base保持序列长度不变）使其保留更多空间细节，利于密集预测任务（如分割）；CNN通过池化逐步压缩空间维度，可能导致细粒度信息丢失。
>
> * **残差连接**：ViT的跳跃连接对特征稳定性和性能影响更大，实验显示移除后精度下降远超CNN（如ResNet），表明其更依赖恒等路径维持梯度流动。
>
> * **表示一致性**：ViT不同深度的特征图间相似性更高，说明其通过注意力迭代细化全局表示，而非CNN的层级特征重构。



**<span style="color: inherit; background-color: rgba(255,246,122,0.8)">关键点：</span>**

* **替代或互补：** ViT并非完全取代CNN，而是在**大数据、强语义关联任务**（如跨模态理解）中展现优势，而CNN在**资源受限、小数据场景**仍具竞争力。

* **趋势融合**：前沿工作（如ConvNeXt、CoAtNet）探索卷积与注意力的混合架构，结合局部效率与全局建模，预示二者界限逐渐模糊。

> - 由于 Transformers 需要大量数据才能达到高精度，因此数据收集过程会延长项目时间。在数据较少的情况下，CNN 的表现通常优于 Transformers。
>
> - Transformer 的训练时间比 CNN 少，如果模型训练时间有限，从计算效率和准确率两个方面比较，可以选择 Transformer。

## 2. ViT模型详解



ViT整体架构可以由上面图中画框的3个模块（5个步骤）组成：

> **Linear Projection of Flattened Patches：**
>
> 获取完整的线性嵌入
>
> * Patch Embedding
>
> * Classification Token
>
> * Positional Embedding

> **Transformer Encoder：**
>
> 提取特征
>
> n \* Transformer Encoder Layer(s)



> **MLP Head：**
>
> 最终的分类层



> 一句话概括：我们将图像分割成固定大小的块，线性嵌入每个块，同时在序列中添加添加一个额外的可学习分类标记cls token（为了进行分类），位置嵌入，并将得到的向量序列输入到标准 Transformer 编码器，得到特征表示。分类任务上，在后面接入MLP利用cls token进行分类。



### 2.1 Patch Embedding

> 作用：通过切分图像块和patch embedding操作得到可以输入到一维向量。



我们用什么样的办法可以将多维的图像数据变成跟NLP一样的一维呢？

> * 按像素展开
>
> 如果直接将图像的每个像素作为一个独立的 patch 进行处理（类似于 NLP 任务中的单个词），那么对于输入尺寸为 224×224 的图像而言，patch 数量将达到  224×224 = 50176。这种方法的计算量过于庞大，难以接受。例如，与 BERT 相比，BERT 仅需处理 512 个 Patch，却已拥有 4810 亿个参数，并且在 2048 块 TPUv4 上训练 20 小时，而该方法的 Patch 数远超 BERT，显然并不可行。
>
> * 按轴展开
>
> 将图像分解为横向和纵向两个独立的一维序列，分别进行自注意力计算，通过分解二维复杂度（H×W）为两次一维计算（H+W），显著降低计算量，但可能忽略对角线方向的全局依赖。
>
> * 用处理过的特征图作为Transformer输入
>
> 先通过卷积神经网络（如 ResNet50）提取特征图，然后将其作为 Transformer 的输入。例如，经过 ResNet50 处理后，图像可被转换为 14×14 的特征图，即 patch 数减少至 196，再将这些 patch 作为输入送入 Transformer，从而降低计算复杂度。
>
> * 将图像划分为多个窗口块
>
> 将图像划分为局部窗口（如7×7），仅在窗口内计算自注意力，类似卷积的局部处理，大幅减少计算开销（复杂度降至HWK²），并通过移位窗口或层级结构实现跨窗口交互（如Swin Transformer）。



#### 2.1.1 Patch 块

> 作用：切分图像块(patch块)

Transformer希望输入一个二维矩阵(N, D)，N是序列的长度，D是序列中每个向量的维度，D常用256。所以如果想要将图像输入到Transformer Encoder中，就需要想办法将$$H\times W\times C$$的三维图像转化为二维输入(N, D)。

$$H \times W \times C \implies N \times (P^{2} \ast C), where N = HW / P^2$$

所以 ViT 利用**等分窗口图片块**的思想，将图像分成块，每个小块称作patch，每个patch块看作NLP Transformer中的一个单词。

将$$H\times W\times C$$的三维图像展平为2D图块序列，序列长度为$$N=HW/P^2$$，每个2D图块的维度为$$P^{2} \ast C$$，其中P是块大小，C是channel数量。

![](<images/ViT(2020, Google Research)-whiteboard_exported_image.png>)

例：Patch块的数据流

原始图片尺寸大小为：$$224 \times 224 \times 3 (H \times W \times C)$$。

每个patch的尺寸设为16（P=16），则每个patch下图片的大小为：$$16 \times 16 \times 3 $$，共有$$(224 / 16) \times (224 / 16) = 14 \times 14 = 196$$个patch。



```python
x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
# 具体是采用了einops库实现
```



#### 2.1.2 Patch Embedding

> 作用：Patch to token
>
> embedding操作将每一个patch块拉伸为一个一维向量，每个一维向量可看作为一个token

在上步patch后，每个patch被展平（flatten）并通过一个**线性变换**（即一个全连接层）映射到一个固定的维度 D，形成每个patch的embedding向量（token）（每个patch对应一个token）。该向量的维度就是Transformer的输入维度。

例：

patch块大小：$$16 \times 16 \times 3 $$，展平后变为$$1 \times 768 $$（$$768 = 16 \times 16 \times 3$$）的token。

共有$$14 \times 14 = 196$$个patch，则得到一个维度为$$196 \times 768$$的矩阵$$X_{patch}$$。

```python
self.patch_to_embedding = nn.Linear(patch_dim, dim)
x = self.patch_to_embedding(x)
```

但后续也对patch embedding方法做了相应的改进：通过2D卷积运算将切分patch块和线性变换合并为一个步骤。

![](<images/ViT(2020, Google Research)-patch_embed.png>)

例：Patch Embedding的数据流

我们将embedding维度embed\_dim设置为768，卷积核大小为$$16 \times 16 \times 3 $$，步长stride设置为16，那么$$224 \times 224 \times 3$$的图像执行卷积运算后，即可得到维度为$$196 \times 768$$的patch embeddings矩阵$$X_{patch}$$。



```python
from functools import partial
 
import torch
import torch.nn as nn
from pyzjr.utils.FormatConver import to_2tuple
 
LayerNorm = partial(nn.LayerNorm, eps=1e-6)
 
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, norm_layer=None):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.embed_dim = embed_dim
        # self.num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
 
    def forward(self, x):
        x = self.proj(x) # 结果形状为 (batch_size, embed_dim, num_patches_H, num_patches_W)
        x = x.flatten(2) # 将输出展平成 (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2) # 转置为 (batch_size, num_patches, embed_dim)
        x = self.norm(x)
        return x
 
 
if __name__ == "__main__":
    img_size = 224  # 图像大小
    patch_size = 16  # 每个patch的大小
    in_channels = 3  # 图像通道数
    embed_dim = 768  # Patch嵌入维度
 
    patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels,
                                     embed_dim=embed_dim)
    batch_size = 2
    x = torch.randn(batch_size, in_channels, img_size, img_size)
    output = patch_embedding(x)
 
    print("Final output shape:", output.shape)
```



> 为什么一定要切分patch，再从patch转为token？
>
> 1. 减少模型计算量
>
> > 在 Transformer 结构中，假设输入序列的长度为 $$N$$，则自注意力机制的计算复杂度为 $$O(N^2)$$，因为每个 token 需要与所有 token（包括自身）计算注意力得分。在 ViT 中，图像的 Patch 数量 $$N=HW/P^2$$ 取决于 Patch 尺寸 $$P$$。当 $$P$$ 过小时，Patch 数量 $$N$$ 增大，导致计算成本急剧上升。因此，我们需要选择合适的 Patch 大小，以平衡模型的计算效率。
>
> * 原图像数据带有较多的冗余信息
>
> > 与自然语言中的丰富语义信息不同，图像数据包含大量冗余信息。相邻像素的取值往往相似，因此我们无需使用过小的 Patch 尺寸（例如 $$P=1$$），因为这样会导致计算粒度过细，增加计算负担，而不会显著提升模型表现。这一特性也是像 MAE（Masked Autoencoder）等基于像素预测的自监督学习模型能够成功的原因之一。



### 2.2 Classification Token

> 作用：在patch embedding上添加(**concat**)一个classification token\[cls]，让 Transformer 在整个图像的上下文中获取全局信息，供下游任务（分类）使用。

Classification Token是参考BERT的思想，在每一个patch embedding前面增加一个\[cls]，通常是放在向量的第一位。\[cls]就是一个类似于“占位符”的向量，表示图像的全局信息。\[cls]是可学习的嵌入向量，它会与其他 patch embeddings 一同输入到 Transformer Encoder 中，经过网络的不断训练，最后模型会学习到\[cls]的输出代表了整个图像的特征，并以该维度的输出来决定最后的输出类别或是其他的下游任务（更直接的说，\[cls]是其他所有patch embeddings对应的图像的类别）。

> 理解方式2：
>
> ViT其实只用到了Transformer的Encoder，而并没有用到Decoder，而 \[cls] 的作用有点类似于解码器中的 Query 的作用，相对应的 Key和Value 就是其他patch embeddings的输出。
>
> 理解方式3：
>
> 其他的 embedding 表达的都是不同的 Patch 的特征，而 \[cls]是要综合所有 Patch 的信息，产生一个新的 embedding 来表达整个图的信息。

![](<images/ViT(2020, Google Research)-cls_token.png>)

例：Classification Token的数据流

\[cls] token是一个大小为$$1 \times 768 $$的向量，我们将其**添加（Concat）**&#x5230;patch embeddings（大小为$$196 \times 768 $$）的前面，那么就得到一个大小为$$197 \times 768 $$的patch embeddings（$$X_{cls+patch}$$）。

进行分类时是取\[cls]对应的embedding进行分类。



```python
# dim=768
self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

# forward前向代码
# 变成(b,196,1024)
cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
# 跟前面的分块进行concat
# 额外追加token，变成b,197,768
x = torch.cat((cls_tokens, x), dim=1)
```



### 2.3 Positional Embedding

> 作用：因为transfomer本身不考虑输入顺序，所以Positional Embedding将位置编码嵌入图像块，通过显式的方式表达patch块在原图的位置信息。

ViT引入了一个positional embedding($$E_{pos}$$)来加入序列的空间位置信息。在ViT中，$$E_{pos}$$是一个**可学习**的向量（允许模型根据数据学习每个位置的语义表示，属于**绝对位置信息**），其长度与$$X_{cls+patch}$$长度一致，所以可以直接相加得到最终的可以输入到Transformer Encoder中的向量。

> 在ViT的研究中发现：位置越接近，往往具有更相似的位置编码。此外，出现了行列结构；同一行/列中的patch具有相似的位置编码。
>
> Positional embedding的生成由两种方式：
>
> * 使用固定算法的Positional embedding
>
> * 使用可学习的Positional embedding
>
> 对于CV的Positional embedding，大致有如下思考：
>
> * 不考虑位置信息
>
> * 将CV当作NLP，只考虑一维位置信息（绝对位置信息）
>
> * 考虑CV特殊的二维位置信息（绝对位置信息）
>
> * 相对位置编码：既考虑相对位置信息又考虑绝对位置信息



例：Positional Embedding的数据流

Positional embedding生成了一个$$197 \times 768 $$的可学习的向量，与大小为$$197 \times 768 $$的$$X_{cls+patch}$$直接相加得到最终的patch embeddings（$$X_{input}$$）

![](<images/ViT(2020, Google Research)-pos_embed.png>)

```python
# num_patches=196，dim=768,+1是因为多了一个cls开启解码标志
self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
```



> $$197 \times 768 $$**的$$X_{input}$$的解释**
>
> * 第一行\[cls]token：
>
> 一个向量，表示整个图像的全局信息，其值会在训练过程中更新，聚合了所有图像块的信息，以便更好地捕捉图像的特征。
>
> * 后196行图像块(patch)特征：
>
> 每一行对应一个图像块的特征，反映该块在图像中的局部信息。这些特征是通过卷积操作提取的，通常采用的是滑动窗口方法，对图像进行分块处理后得到。
>
> * 位置嵌入



### 2.4 Transformer Encoder

> 作用：根据输入的向量信息，提取特征

Transformer Encoder的输入：**`Input = Patch Embedding + Classification token + Positional Embedding`**

$$X_{input}=X_{cls+patch}+E_{pos}$$

Transformer Encoder是两个块的堆叠，然后再整体叠加 L(/n) 次（在ViT-Base的架构中，L=12，12个Encoder Block）。这两个块指的是：

* LayerNorm + Multi-Head Attention

* LayerNorm + MLP

![](<images/ViT(2020, Google Research)-image-1.png>)

> **关键点**
>
> 1. **Transformer Encoder的输入shape和输出shape是保持不变的。**
>
> 2. 从实现代码中可以看出，**在整个Transformer Encoder之后还接了一个LayerNorm层。**
>
> 3. LayerNorm
>
> 虽然在 CV 里用的比较多的是 BatchNorm，但是在ViT中仍然是用与NLP相同的LN。
>
> <span style="color: inherit; background-color: rgb(251,191,188)">与NLP使用LN而不使用BN相同的答案：</span>BN是对每个通道的所有样本（样本间）进行归一化，而LN是对每个样本（样本内）的所有特征进行归一化。<span style="color: rgb(36,91,219); background-color: inherit">更具体到ViT中，虽然LN处理的是图片数据，但在进行LN之前，图片已经被切割成了Patch，而每个Patch表示的是一个词，因此是在用语义的逻辑在解决视觉问题，因此在ViT中，LN也是按语义的逻辑在用的。</span>
>
> * Multi-Head Attention
>
> 目的：为了使网络能够综合利用多方面角度提取更加准确的表示，从而可以捕捉到更加丰富的特征，**可以类比 CNN 中多个核分别提取特征的作用**。
>
> * 在ViT中，Attention具体学到了什么
>
> 浅层网络中，ViT还只能关注到距离较近的像素点；随着网络加深，ViT逐渐学会去更远的像素点中寻找相关信息。这个过程就和用在CNN中用卷积逐层去扩大感受野非常相似。

MLP block : 由两个Linear层（全连接层）+GeLU激活函数+Dropout层组成，采用的是倒瓶颈结构，具体结构可以看下面的实现代码。其中第一个全连接层会将输入节点个数翻4倍：$$197 \times 768 $$变为$$197 \times 3072 $$，第二个全连接层会还原节点个数$$197 \times 3072 $$变为$$197 \times 768 $$。

> **瓶颈结构（Bottleneck）**（如 ResNet）：**先降维再升维**
>
> * 降维（减少通道数） → 卷积操作 → 升维（恢复通道数）
>
> * 目的是降低计算量，同时保证网络的表达能力。
>
> **倒瓶颈结构（Inverted Bottleneck）**（如 MobileNetV2）：**先升维再降维**
>
> * 升维（增加通道数） → 非线性变换（如深度可分离卷积或 MLP 操作） → 降维（恢复原始通道数）
>
> * 目的是在高维空间中进行复杂特征变换，从而增强特征表达能力。

![](<images/ViT(2020, Google Research)-encoder.png>)

例：Transformer Encoder的数据流

Transformer Encoder中第一个Encoder Block接受$$197 \times 768 $$的矩阵输入。对于后续所有Encoder Block，输入都是来自前一个Encoder Block的大小为$$197 \times 768 $$的输出矩阵。

在一个Encoder Block的内部，输入首先通过LayerNorm，维度不变，然后输入到MultiHead Attention中。

在MultiHead Attention中，使用Linear层将输入转化为$$197 \times 2304（768 \times 3）$$的qkv矩阵。然后将qkv矩阵重塑为$$197\times 3 \times 768 $$三个矩阵，分别代表q、k、v矩阵（大小为$$197 \times 768 $$）。然后q、k、v再进一步重塑为$$12\times 197 \times 64 $$大小的矩阵，代表12个注意力头。然后可以进行注意力计算。

将MultiHead Attention的输出，添加到输入（残差连接）获得最终的输出，然后进行后续的LayerNorm、MLP等。



```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    """
    MultiHead Attention
    qkv矩阵使用线性层获取
    """
    def __init__(
            self,
            embed_dim,               # 输入token的嵌入维度
            num_heads=8,             # 注意力头的数量
            qkv_bias=False,          # 是否在qkv计算中添加偏置项
            attention_dropout_ratio=0.,  # 注意力权重的dropout比例
            proj_drop=0.,            # 输出投影后的dropout比例
    ):
        super(MultiheadAttention, self).__init__()
        # 初始化参数
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 每个注意力头的维度
        self.scale = self.head_dim ** -0.5      # 缩放因子，用于缩放点积结果
        
        # 使用一个线性层同时生成q、k、v，提高计算效率
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        
        # 定义注意力dropout层和输出投影层
        self.attn_drop = nn.Dropout(attention_dropout_ratio)
        self.out_linear = nn.Linear(embed_dim, embed_dim)  # 输出投影
        self.out_linear_drop = nn.Dropout(proj_drop)        # 输出dropout

    def forward(self, x):
        # 输入x的形状：[batch_size, num_patches + 1, embed_dim]
        B, N, C = x.shape
        
        # 生成qkv张量并重塑形状为多头形式
        # qkv的形状：[batch_size, num_patches + 1, 3, num_heads, head_dim]
        # 经过permute后变为：[3, batch_size, num_heads, num_patches + 1, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        # 将qkv拆分为独立的q、k、v张量
        # 每个张量的形状：[batch_size, num_heads, num_patches + 1, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力矩阵（点积注意力）
        # attn形状：[batch_size, num_heads, num_patches+1, num_patches+1]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 转置最后两个维度并缩放
        
        # 对注意力矩阵进行softmax归一化
        attn = F.softmax(attn, dim=-1)
        
        # 对注意力权重进行dropout
        attn = self.attn_drop(attn)
        
        # 将注意力权重应用到value上，并重塑形状
        # 结果形状：[batch_size, num_patches + 1, embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # 合并多头
        
        # 输出投影和dropout
        x = self.out_linear(x)
        x = self.out_linear_drop(x)
        return x

class MLP(nn.Module):
    """
    MLP block由两个线性层和GeLU激活层组成
    """
    def __init__(
            self,
            in_features,              # 输入特征维度
            hidden_features=None,     # 隐藏层特征维度（默认是输入的4倍）
            out_features=None,        # 输出特征维度（默认与输入相同）
            act_layer=nn.GELU,        # 激活函数类型（默认GELU）
            drop_ratio=0.             # dropout比例
    ):
        super().__init__()
        # 设置输出维度，默认与输入相同
        out_features = out_features or in_features
        # 设置隐藏层维度，默认是输入维度的4倍（ViT标准设计）
        hidden_features = hidden_features or in_features
        
        # 定义第一个全连接层（扩展维度）
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数
        self.act = act_layer()
        # 定义第二个全连接层（压缩回原维度）
        self.fc2 = nn.Linear(hidden_features, out_features)
        # dropout层
        self.drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        # 前向传播过程
        x = self.fc1(x)      # 扩展维度
        x = self.act(x)       # 激活函数
        x = self.drop(x)      # dropout
        x = self.fc2(x)      # 压缩回原维度
        x = self.drop(x)      # dropout
        return x

class EncoderBlock(nn.Module):
    """
    Transformer Encoder Block
    一个单层Block由LayerNorm、Attention和MLP组成。
    在Mlp block中, MLP层的隐藏维度是输入的维度的4倍,
    """
    mlp_ratio = 4  # MLP隐藏层维度扩展比例（ViT标准设计）

    def __init__(
            self,
            dim,                     # 输入token的维度
            num_heads,               # 注意力头数量
            qkv_bias=False,          # 是否在qkv中使用偏置
            drop_ratio=0.,           # 投影后的dropout比例
            attention_dropout_ratio=0.,  # 注意力权重的dropout比例
            drop_path_ratio=0.,       # DropPath的比例（用于Stochastic Depth）
            norm_layer=nn.LayerNorm,  # 标准化层类型（默认LayerNorm）
            act_layer=nn.GELU         # 激活函数类型（默认GELU）
    ):
        super(EncoderBlock, self).__init__()
        # 第一个标准化层（在注意力之前）
        self.norm1 = norm_layer(dim)
        
        # 多头注意力模块
        self.attention = MultiheadAttention(
            dim, num_heads,
            qkv_bias=qkv_bias,
            attention_dropout_ratio=attention_dropout_ratio,
            proj_drop=drop_ratio
        )
        
        # Stochastic Depth模块（如果比例为0则使用恒等映射）
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        
        # 第二个标准化层（在MLP之前）
        self.norm2 = norm_layer(dim)
        
        # 计算MLP隐藏层维度（输入维度的4倍）
        mlp_hidden_dim = int(dim * self.mlp_ratio)
        
        # MLP模块
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            drop_ratio=drop_ratio,
            act_layer=act_layer
        )

    def forward(self, x):
        # 残差连接1：多头注意力部分
        # 先标准化，再做注意力，然后DropPath，最后加残差
        x = x + self.drop_path(self.attention(self.norm1(x)))
        
        # 残差连接2：MLP部分
        # 先标准化，再做MLP，然后DropPath，最后加残差
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    """
    堆叠 L 次 Transformer Encoder Block
    组成完整的Transformer Encoder
    """
    def __init__(
            self,
            num_layers,              # 编码器层数（L）
            dim,                     # token维度
            num_heads,               # 注意力头数量
            qkv_bias=False,          # 是否使用qkv偏置
            drop_ratio=0.,           # 投影dropout比例
            attention_dropout_ratio=0.,  # 注意力权重dropout比例
            drop_path_ratio=0.,       # 最大DropPath比例
            norm_layer=nn.LayerNorm,  # 标准化层
            act_layer=nn.GELU         # 激活函数
    ):
        super(TransformerEncoder, self).__init__()
        # 生成DropPath比率列表（线性递增）
        # 例如：当num_layers=3，drop_path_ratio=0.3时，生成[0.0, 0.15, 0.3]
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, num_layers)]
        
        # 创建多层编码器块
        self.layers = nn.ModuleList([
            EncoderBlock(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop_ratio=drop_ratio,
                attention_dropout_ratio=attention_dropout_ratio,
                drop_path_ratio=dpr[_],  # 每层使用不同的DropPath比率
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for _ in range(num_layers)
        ])
        
        # 最终的标准化层
        self.norm = norm_layer(dim)

    def forward(self, x):
        # 逐层通过所有编码器块
        for layer in self.layers:
            x = layer(x)
        
        # 应用最终的标准化
        x = self.norm(x) # 注意：在所有Encoder Block之后会有一个LayerNorm层
        return x
```



### 2.5 MLP Head

> <span style="color: rgb(143,149,158); background-color: inherit">作用：用非线性激活函数去做分类的预测</span>

MLP Head 通常包括以下几层：

* 全连接层（Linear）：将 768 维特征向量映射到类别数（如 1000 个类别），用线性变换处理。

* 激活函数：可能使用激活函数（如 ReLU 或 GELU），引入非线性。

* 输出层：另一个全连接层，最终输出分类 logits（未归一化的分类分数）。

![](<images/ViT(2020, Google Research)-MLPHead.png>)

例：

从Transformer Encoder中输出$$197 \times 768 $$的向量，从该向量中提取出$$1 \times 768 $$的\[cls]用于MLP进行分类。

在原论文中，作者说训练ImageNet-21K时MLP Head是由`Linear+tanh激活函数+Linear`组成。但是迁移到ImageNet-1K时或者自己的数据上时，用一个`Linear`也可以。



> ViT主类的代码

```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    '''
    ViT主类
    '''
    def __init__(
            self,
            img_size=224,                # 输入图像尺寸（标准ViT为224x224）
            patch_size=16,               # 图像分块大小（16x16像素为一个patch）
            in_channels=3,               # 输入通道数（RGB图像为3）
            num_classes=1000,            # 分类类别数（ImageNet为1000）
            hidden_dim=768,              # 嵌入维度（每个patch的编码维度）
            num_heads=12,                # 多头注意力的头数
            num_layers=12,               # Transformer编码器层数（ViT-Base为12层）
            qkv_bias=True,              # 是否在QKV计算中使用偏置项
            drop_ratio=0.,              # 通用dropout比例
            attention_dropout_ratio=0.,  # 注意力权重的dropout比例
            drop_path_ratio=0.,          # Stochastic Depth最大丢弃比例
            norm_layer=LayerNorm,        # 标准化层类型（默认LayerNorm）
            act_layer=nn.GELU            # 激活函数类型（默认GELU）
    ):
        super(VisionTransformer, self).__init__()
        # 验证输入图像尺寸合法性
        assert img_size == 224, f"Image size must be 224, but got {img_size}"
        assert img_size % patch_size == 0, f"Image size {img_size} must be divisible by patch size {patch_size}"
        
        # 基础参数设置
        self.num_classes = num_classes
        self.num_tokens = 1  # 分类token的数量（ViT中固定为1个CLS token）
        
        # 图像分块嵌入层：将图像转换为patch序列
        # 输出形状：[B, num_patches, hidden_dim]
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_dim,
            norm_layer=norm_layer
        )
        
        # 计算patch数量（例如224/16=14 -> 14x14=196个patches）
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        
        # 可学习的分类token（CLS token）：用于最终分类的特征
        # 初始化为全零，形状：[1, 1, hidden_dim]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # 可学习的位置编码：为每个patch+CLS token添加位置信息
        # 形状：[1, num_patches + 1, hidden_dim]（+1是CLS token的位置）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, hidden_dim))
        
        # 位置编码后的Dropout层
        self.pos_drop = nn.Dropout(p=drop_ratio)
        
        # Transformer编码器堆叠（核心处理模块）
        self.blocks = TransformerEncoder(
            num_layers=num_layers,
            dim=hidden_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_ratio=drop_ratio,
            attention_dropout_ratio=attention_dropout_ratio,
            drop_path_ratio=drop_path_ratio,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        
        # 最终的标准化层
        self.norm = norm_layer(hidden_dim)
        
        # 分类头：将CLS token特征映射到类别空间
        self.head = nn.Linear(hidden_dim, num_classes)
        
        # 初始化模型权重
        self._initialize_weights()

    # 权重初始化方法
    def _initialize_weights(self):
        # 遍历所有子模块
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 线性层：截断正态分布初始化权重（标准差0.01），偏置置零
                nn.init.trunc_normal_(m.weight, std=.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                # 卷积层：Kaiming正态分布初始化（用于patch embedding）
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm层：权重初始化为1，偏置置零
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    # 前向传播过程
    def forward(self, x):
        # 输入x形状：[B, C, H, W]（如[B,3,224,224]）
        
        # Step 1: 将图像转换为patch embeddings
        x = self.patch_embed(x)  # 输出形状：[B, 196, 768]（假设hidden_dim=768）
        
        # Step 2: 扩展CLS token到当前batch size
        # 原始形状：[1,1,768] → 扩展后：[B,1,768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        
        # Step 3: 将CLS token拼接到patch序列前
        # 拼接后形状：[B, 196+1, 768] → [B,197,768]
        x = torch.cat((cls_token, x), dim=1)
        
        # Step 4: 添加位置编码并应用Dropout
        # pos_embed形状自动广播到[B,197,768]
        x = self.pos_drop(x + self.pos_embed)
        
        # Step 5: 通过Transformer编码器堆叠
        # 输出形状保持：[B,197,768]
        x = self.blocks(x)
        
        # Step 6: 只取CLS token的特征（索引0的位置）
        # 输出形状：[B,768]
        x = x[:, 0]  # 等同于x[:, 0, :]
        
        # Step 7: 通过分类头得到最终logits
        # 输出形状：[B, num_classes]（如[B,1000]）
        x = self.head(x)
        
        return x
```



### 2.6 ViT的工作流程

官方动态工作流程：

![](<images/ViT(2020, Google Research)-1.gif>)

整合2.1到2.5节中的ViT图示：

![](<images/ViT(2020, Google Research)-ViT_图示_中.png>)

具体流程图如下：

![](<images/ViT(2020, Google Research)-vit流程图.png>)

Transformer Ecoder中的数据流：

[Aman Arora's blog](https://amaarora.github.io/posts/2021-01-18-ViT.html)中有比较清楚的数据流图，如下所示：

![](<images/ViT(2020, Google Research)-image-2.png>)

动态流动图可以参考：



## 3. ViT训练与微调

训练方法：通常采用 **<span style="color: inherit; background-color: rgb(251,191,188)">大规模预训练 + 迁移学习（微调）</span>** 的策略，以充分利用大规模数据集的学习能力，并适应不同任务的数据需求。

> 预训练模型使用到的数据集：
>
> * ILSVRC-2012 ImageNet dataset：1000 classes
>
> * ImageNet-21k：21k classes
>
> * JFT：18k High Resolution Images

> 将预训练迁移到小数据集上：
>
> * CIFAR-10/100
>
> * Oxford-IIIT Pets
>
> * Oxford Flowers-102
>
> * VTAB



原论文的作者设计了3种不同大小的ViT模型：

| DModel    | Layers | Hidden Size | MLP Size | Heads | Params |
| --------- | ------ | ----------- | -------- | ----- | ------ |
| ViT-Base  | 12     | 768         | 3072     | 12    | 86M    |
| ViT-Large | 24     | 1024        | 4096     | 16    | 307M   |
| ViT-Huge  | 32     | 1280        | 5120     | 16    | 632M   |

> Layers：Transformer Encoder中重复堆叠Encoder Block的次数
>
> Hidden Size：对应通过Embedding层后每个token的dim（向量的长度）
>
> MLP Size：Transformer Encoder中MLP Block第一个全连接的节点个数（是Hidden Size的四倍）
>
> Heads：Transformer中Multi-Head Attention的heads数

ViT-L/16代表`ViT-Large + 16 patch size`

### 3.1 预训练阶段（Pre-training）

1. 目标：
   在 **超大规模数据集**（如 JFT-300M、ImageNet-21K）上预训练 ViT，以学习通用的视觉特征。

2. 步骤：

   * 直接训练 ViT

   * 调整 Prediction Head：

     * 预训练时使用 **线性分类头（Prediction Head）** 进行类别预测。

     * 预训练完成后，在迁移学习时，需要替换为新的 $$D \times K$$ 前馈层（Feed Forward Layer），其中：

       * $$D$$ 是 Transformer 输出的维度。

       * $$K$$ 是目标数据集的类别数。

3. 优化策略：

   * 使用 **AdamW** 作为优化器，提高训练稳定性。

   * 采&#x7528;**&#x20;学习率衰减（Learning Rate Decay）** &#x548C;**&#x20;学习率热身（Warm-up）&#x20;**&#x7B56;略，提高训练效果。

   * 应用 **Stochastic Depth（随机深度丢弃**），避免梯度消失。



### 3.2 迁移学习/微调阶段（Fine-tuning）

> 经过预训练的 ViT 模型是一种强大的特征提取器，我们可以利用其输出的特征来完成更多丰富的下游任务，如更复杂的分类任务、目标检测等。在执行这些任务时，我们会向预训练模型输入新的数据，同时尽可能保持模型的主体架构不变。例如，固定 ViT 的整体参数，仅在输出层后添加一个新的模型，并在训练过程中仅更新该新模型的参数。这种方法既能充分利用 ViT 预训练模型的特征提取能力，又能有效适应不同的任务需求。

1. 目标：
   将预训练的 ViT 迁移到较小规模的下游数据集（如 ImageNet-1K、CIFAR-100），适应特定任务。

2. 关键做法：

   * 调整分类头：

     * 去掉原来的 prediction head，替换为一个适应新数据集的前馈层（$$D \times K$$）。

   * 适应不同分辨率的输入：

     * **Patch 大小保持不变：**&#x5982;果输入图片尺寸更大，则 Patch 数$$N=HW/P^2$$ 增加。

     * **Positional Encoding 需要调整**：

       * ViT 的位置编码（Positional Encoding）最初是针对预训练输入的固定尺寸设计的。

       * 若输入图像尺寸不同，位置编码不能直接复用，需要通过 **2D 插值（双线性插值Bilinear Interpolation）** 来适配新的 Patch 数，以保持空间信息一致性。



## 4. ViT的发展

ViT 论证了大规模+大模型在图像领域的涌现能力，也**为后续多模态任务的发展奠定了基石。**

ViT的突出应用：

图像分类、图像标题生成、图像分割、异常检测、动作识别、自动驾驶等。

![](<images/ViT(2020, Google Research)-image-3.png>)



