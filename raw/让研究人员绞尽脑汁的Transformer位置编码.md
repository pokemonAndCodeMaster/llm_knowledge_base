# 让研究人员绞尽脑汁的Transformer位置编码
**作者:** 苏剑林 (科学空间)
**日期:** 2021-02-03
**原文链接:** https://kexue.fm/archives/8130

不同于RNN、CNN等模型，对于Transformer模型来说，位置编码的加入是必不可少的，因为纯粹的Attention模块是无法捕捉输入顺序的，即无法区分不同位置的Token。为此我们大体有两个选择：1、想办法将位置信息融入到输入中，这构成了绝对位置编码的一般做法；2、想办法微调一下Attention结构，使得它有能力分辨不同位置的Token，这构成了相对位置编码的一般做法。

## 绝对位置编码
形式上来看，绝对位置编码是相对简单的一种方案，但即便如此，也不妨碍各路研究人员的奇思妙想，也有不少的变种。一般来说，绝对位置编码会加到输入中：在输入的第$k$个向量$x_k$中加入位置向量$p_k$变为$x_k+p_k$，其中$p_k$只依赖于位置编号$k$。

### 训练式
很显然，绝对位置编码的一个最朴素方案是不特意去设计什么，而是直接将位置编码当作可训练参数，比如最大长度为512，编码维度为768，那么就初始化一个$512 	imes 768$的矩阵作为位置向量，让它随着训练过程更新。现在的BERT、GPT等模型所用的就是这种位置编码，事实上它还可以追溯得更早，比如2017年Facebook的《Convolutional Sequence to Sequence Learning》就已经用到了它。

对于这种训练式的绝对位置编码，一般的认为它的缺点是没有外推性，即如果预训练最大长度为512的话，那么最多就只能处理长度为512的句子，再长就处理不了了。当然，也可以将超过512的位置向量随机初始化，然后继续微调。但笔者最近的研究表明，通过层次分解的方式，可以使得绝对位置编码能外推到足够长的范围，同时保持还不错的效果，细节请参考笔者之前的博文《层次分解位置编码，让BERT可以处理超长文本》。因此，其实外推性也不是绝对位置编码的明显缺点。

### 三角式
三角函数式位置编码，一般也称为Sinusoidal位置编码，是Google的论文《Attention is All You Need》所提出来的一个显式解：
$$
\begin{cases}
p_{k,2i} = \sin(k/10000^{2i/d}) 
p_{k,2i+1} = \cos(k/10000^{2i/d})
\end{cases}
$$
其中$p_{k,2i}, p_{k,2i+1}$分别是位置$k$的编码向量的第$2i, 2i+1$个分量，$d$是位置向量的维度。

很明显，三角函数式位置编码的特点是有显式的生成规律，因此可以期望于它有一定的外推性。另外一个使用它的理由是：由于$\sin(\alpha+\beta)=\sin\alpha\cos\beta+\cos\alpha\sin\beta$以及$\cos(\alpha+\beta)=\cos\alpha\cos\beta-\sin\alpha\sin\beta$，这表明位置$\alpha+\beta$的向量可以表示成位置$\alpha$和位置$\beta$的向量组合，这提供了表达相对位置信息的可能性。但很奇怪的是，现在我们很少能看到直接使用这种形式的绝对位置编码的工作，原因不详。

### 递归式
原则上来说，RNN模型不需要位置编码，它在结构上就自带了学习到位置信息的可能性（因为递归就意味着我们可以训练一个“数数”模型），因此，如果在输入后面先接一层RNN，然后再接Transformer，那么理论上就不需要加位置编码了。同理，我们也可以用RNN模型来学习一种绝对位置编码，比如从一个向量$p_0$出发，通过递归格式$p_{k+1}=f(p_k)$来得到各个位置的编码向量。

ICML 2020的论文《Learning to Encode Position for Transformer with Continuous Dynamical Model》把这个思想推到了极致，它提出了用微分方程（ODE）$dp_t/dt=h(p_t,t)$的方式来建模位置编码，该方案称之为FLOATER。显然，FLOATER也属于递归模型，函数$h(p_t,t)$可以通过神经网络来建模，因此这种微分方程也称为神经微分方程，关于它的工作最近也逐渐多了起来。

理论上来说，基于递归模型的位置编码也具有比较好的外推性，同时它也比三角函数式的位置编码有更好的灵活性（比如容易证明三角函数式的位置编码就是FLOATER的某个特解）。但是很明显，递归形式的位置编码牺牲了一定的并行性，可能会带速度瓶颈。

### 相乘式
刚才我们说到，输入$x_k$与绝对位置编码$p_k$的组合方式一般是$x_k+p_k$，那有没有“不一般”的组合方式呢？比如$x_k \otimes p_k$（逐位相乘）？我们平时在搭建模型的时候，对于融合两个向量有多种方式，相加、相乘甚至拼接都是可以考虑的，怎么大家在做绝对位置编码的时候，都默认只考虑相加了？

很抱歉，笔者也不知道答案。可能大家默认选择相加是因为向量的相加具有比较鲜明的几何意义，但是对于深度学习模型来说，这种几何意义其实没有什么实际的价值。最近笔者看到的一个实验显示，似乎将“加”换成“乘”，也就是$x_k \otimes p_k$的方式，似乎比$x_k+p_k$能取得更好的结果。具体效果笔者也没有完整对比过，只是提供这么一种可能性。关于实验来源，可以参考《中文语言模型研究：(1) 乘性位置编码》。

## 相对位置编码
相对位置并没有完整建模每个输入的位置信息，而是在算Attention的时候考虑当前位置与被Attention的位置的相对距离，由于自然语言一般更依赖于相对位置，所以相对位置编码通常也有着优秀的表现。对于相对位置编码来说，它的灵活性更大，更加体现出了研究人员的“天马行空”。

### 经典式
相对位置编码起源于Google的论文《Self-Attention with Relative Position Representations》，华为开源的NEZHA模型也用到这种位置编码，后面各种相对位置编码变体基本也是依葫芦画瓢的简单修改。

一般认为，相对位置编码是由绝对位置编码启发而来，考虑一般的带绝对位置编码的Attention：
$$
\begin{cases}
q_i = (x_i+p_i)W_Q 
k_j = (x_j+p_j)W_K 
v_j = (x_j+p_j)W_V 
a_{i,j} = 	ext{softmax}(q_i k_j^	op) 
o_i = \sum_j a_{i,j} v_j
\end{cases}
$$
其中softmax对$j$那一维归一化，这里的向量都是指行向量。我们初步展开$q_i k_j^	op$：
$$
q_i k_j^	op = (x_iW_Q+p_iW_Q)(W_K^	op x_j^	op+W_K^	op p_j^	op)
$$
为了引入相对位置信息，Google把第一项位置去掉，第二项$p_jW_K$改为二元位置向量$R_{K_{i,j}}$，变成
$$
a_{i,j} = 	ext{softmax}(x_iW_Q(x_jW_K+R_{K_{i,j}})^	op)
$$
以及$o_i=\sum_j a_{i,j}v_j=\sum_j a_{i,j}(x_jW_V+p_jW_V)$中的$p_jW_V$换成$R_{V_{i,j}}$：
$$
o_i = \sum_j a_{i,j}(x_jW_V+R_{V_{i,j}})
$$
所谓相对位置，是将本来依赖于二元坐标$(i,j)$的向量$R_{K_{i,j}}, R_{V_{i,j}}$，改为只依赖于相对距离$i-j$，并且通常来说会进行截断，以适应不同任意的距离
$$
\begin{cases}
R_{K_{i,j}} = p_K[	ext{clip}(i-j, p_{\min}, p_{\max})] 
R_{V_{i,j}} = p_V[	ext{clip}(i-j, p_{\min}, p_{\max})]
\end{cases}
$$
这样一来，只需要有限个位置编码，就可以表达出任意长度的相对位置（因为进行了截断），不管$p_K, p_V$是选择可训练式的还是三角函数式的，都可以达到处理任意长度文本的需求。

### XLNET式
XLNET式位置编码其实源自Transformer-XL的论文《Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context》，只不过因为使用了Transformer-XL架构的XLNET模型并在一定程度上超过了BERT后，Transformer-XL才算广为人知，因此这种位置编码通常也被冠以XLNET之名。

XLNET式位置编码源于对上述$q_i k_j^	op$的完全展开：
$$
q_i k_j^	op = x_iW_QW_K^	op x_j^	op + x_iW_QW_K^	op p_j^	op + p_iW_QW_K^	op x_j^	op + p_iW_QW_K^	op p_j^	op
$$
Transformer-XL的做法很简单，直接将$p_j$替换为相对位置向量$R_{i-j}$，至于两个$p_i$，则干脆替换为两个可训练的向量$u,v$：
$$
x_iW_QW_K^	op x_j^	op + x_iW_QW_K^	op R_{i-j}^	op + uW_QW_K^	op x_j^	op + vW_QW_K^	op R_{i-j}^	op
$$
该编码方式中的$R_{i-j}$没有像式(6)那样进行截断，而是直接用了Sinusoidal式的生成方案，由于$R_{i-j}$的编码空间与$x_j$不一定相同，所以$R_{i-j}$前面的$W_K^	op$换了另一个独立的矩阵$W_{K,R}^	op$，还有$uW_Q 、vW_Q$可以直接合并为单个$u 、v$，所以最终使用的式子是
$$
x_iW_QW_K^	op x_j^	op + x_iW_QW_{K,R}^	op R_{i-j}^	op + uW_K^	op x_j^	op + vW_{K,R}^	op R_{i-j}^	op
$$
此外，$v_j$上的位置偏置就直接去掉了，即直接令$o_i=\sum_j a_{i,j}x_jW_V$。似乎从这个工作开始，后面的相对位置编码都只加到Attention矩阵上去，而不加到$v_j$上去了。

### T5式
T5模型出自文章《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》，里边用到了一种更简单的相对位置编码。思路依然源自展开式(7)，如果非要分析每一项的含义，那么可以分别理解为“输入-输入”、“输入-位置”、“位置-输入”、“位置-位置”四项注意力的组合。如果我们认为输入信息与位置信息应该是独立（解耦）的，那么它们就不应该有过多的交互，所以“输入-位置”、“位置-输入”两项Attention可以删掉，而$p_iW_QW_K^	op p_j^	op$实际上只是一个只依赖于$(i,j)$的标量，我们可以直接将它作为参数训练出来，即简化为
$$
x_iW_QW_K^	op x_j^	op + \beta_{i,j}
$$
说白了，它仅仅是在Attention矩阵的基础上加一个可训练的偏置项而已，而跟XLNET式一样，在$v_j$上的位置偏置则直接被去掉了。包含同样的思想的还有微软在ICLR 2021的论文《Rethinking Positional Encoding in Language Pre-training》中提出的TUPE位置编码。

比较“别致”的是，不同于常规位置编码对将$\beta_{i,j}$视为$i-j$的函数并进行截断的做法，T5对相对位置进行了一个“分桶”处理，即相对位置是$i-j$的位置实际上对应的是$f(i-j)$位置，具体的映射关系涉及多级映射。

### DeBERTa式
DeBERTa主要改进也是在位置编码上，同样还是从展开式(7)出发，T5是干脆去掉了第2、3项，只保留第4项并替换为相对位置编码，而DeBERTa则刚刚相反，它扔掉了第4项，保留第2、3项并且替换为相对位置编码：
$$
q_i k_j^	op = x_iW_QW_K^	op x_j^	op + x_iW_QW_K^	op R_{i,j}^	op + R_{j,i}W_QW_K^	op x_j^	op
$$
至于$R_{i,j}$的设计也是像式(6)那样进行截断的。此外，DeBERTa还提出了增强遮罩解码（EMD）的概念。

## 其他位置编码

### CNN式
CNN模型的位置信息可能是由 Zero Padding 泄漏的。该操作导致模型有能力识别位置信息，实际上提取的是当前位置与padding的边界的相对距离。

### 复数式 (Complex Order)
来自ICLR 2020的论文《Encoding word order in complex embeddings》。将位置编码定义在复数空间：
$$
[r_{j,1}e^{i(\omega_{j,1}k+	heta_{j,1})}, \dots, r_{j,d}e^{i(\omega_{j,d}k+	heta_{j,d})}]
$$
这种方案将整个 Transformer 模型都改为了复数版。

### 融合式 (融绝对与相对为一体)
作者提出的一种基于复数乘法的方案：
两个二维向量的内积，等于把它们当复数看时，一个复数与另一个复数的共轭的乘积实部。
$$
\langle q_m e^{im	heta}, k_n e^{in	heta} angle = 	ext{Re}[q_m k_n^* e^{i(m-n)	heta}]
$$
这巧妙地将绝对位置与相对位置融合在一起了。这种思想后来演化为了旋转位置编码 (RoPE)。
