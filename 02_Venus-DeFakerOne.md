# 02 · Venus-DeFakerOne: Unified Fake Image Detection & Localization

## 0. 元信息

| 项目 | 内容 |
|---|---|
| 作者 | GuangJianTeam, AntGroup（蚂蚁集团光鉴团队）。贡献者名单见附录 C：Changjiang Jiang、Chenfan Qu（数据负责人）、Jiangwei Xie、Mingqi Fang、Song Zhou、Xuekang Zhu、Chenfeng Zhang、Longfei Liu、Zhenming Wang、Jian Liu、Jingjing Liu（项目负责人）、Weiqiang Wang（项目顾问） |
| 发表 | arXiv 预印本，arXiv:2605.14091v2，日期 2026 年 5 月 15 日 |
| 代码 | https://github.com/venus-guangjian/Venus-DeFakerOne |
| 篇幅 | 正文 21 页 + 参考文献 + 附录（相关工作、数据构成、鲁棒性明细），共 35 页 |

> 注意：这是一篇**工业界长文预印本**，不是会议论文。数据体系中大量为私有数据，这一点在阅读和引用时必须牢记。

---

## 1. 一句话总结

用**12.5M 多域训练数据** + InternVL2-2B + SAM2，把原本碎片化的四个取证域（Document / Nature / DeepFake / AIGC）统一到一个检测+定位的基础模型里，并用一整套数据配比实验给出「在 FIDL 里，数据组成比数据规模更重要」的经验规律。

**贡献重心在数据与经验规律，不在网络结构**——网络是已有组件的组合（InternVL2 + SAM2 + SA2VA 式架构）。

---

## 2. 动机：为什么现在需要「统一 FIDL」

### 2.1 从碎片化伪造到统一伪造

论文 Figure 2 的对比是理解全篇的入口：

| | 过去（Fragmented） | 现在（Unified） |
|---|---|---|
| 域划分 | Document / Nature / DeepFake / AIGC 各自独立 | 基础生成模型打破域边界 |
| 典型操作 | 文本替换插入、拼接、复制粘贴、人脸替换 | T2I / I2I 生成与编辑，覆盖上述全部 |
| 伪影假设 | 各域有专属伪影 → 域专用方法有效 | 伪影跨域**可迁移且相互纠缠** |
| 防御难度 | 容易 | 困难 |

具体到各域的传统假设：

- **Document**：关注文字、版面与局部伪影（手工编辑或 AnyText 一类的字形引导生成）；
- **Nature**：关注区域不一致性与像素级定位；
- **DeepFake**：关注摩尔纹、边界伪影与生理不一致；
- **AIGC**：关注全局统计分布偏移与生成伪影。

当 GPT-Image-2 这类基础生成模型同时具备「开域生成 + 图像编辑」能力时，传统依赖域专用伪影的检测器就会失效。

### 2.2 两大挑战（论文明确列出的）

1. **缺乏对多域伪影交互的系统建模**：不同子领域共享输入输出形式，但取证监督粒度与底层伪影不同。多域数据是否相互**可迁移**、还是相互**冲突**，此前尚未被充分探索。
2. **缺乏对大容量统一模型的研究**：现有方法仍以小视觉模型范式为主，围绕特定伪影设计，特征空间容量不足以表征跨四域的多样伪造痕迹，也难以在同一框架里同时做好图像级检测、像素级定位与跨域泛化。

---

## 3. 数据构造（本文最重要的部分）

### 3.1 两阶段数据策略

| 阶段 | 规模 | 组成 | 目的 |
|---|---|---|---|
| **Stage-1 范式验证** | **2M** | 75% 私有业务数据（文档为主）+ 25% API 生成与开源基准 | 验证模型能在四域多任务上有效收敛、不出现灾难性干扰 |
| **Stage-2 能力扩展与规模化** | **12.5M** | 五条互补管线：开源数据集、私有业务数据、API 生成、高质量专家 PSD 合成、内部红队对抗样本 | 统一 FIDL 训练，并驱动数据 scaling 实验 |

Stage-2 采用**三层数据架构**：真实（authentic）→ 合成（synthetic）→ 对抗（adversarial）。

### 3.2 12.5M 的域构成（附录 Table 11）

| 域 | 合计 | 公开部分 | 私有部分 |
|---|---|---|---|
| DeepFake | 3.1M | 0.88M（FF++、CelebDF-v2、DFD、DFDC、ScaleDF、DF40、WDF、MFFI） | **2.22M** |
| AIGC | 3.6M | 2.344M（DiffusionForensics、CommunityForensics、GenImage、LAION_DATA、ForenSynths） | **1.256M** |
| Document | 2.5M | 0.162M（DocTamper、T-SROIE、RTM、SACP、RIFLC、OSTF） | **2.338M** |
| Nature | 3.3M | 3.3M（MIML、CASIA-v2、COCO_2017、OpenSDI、So-Fake-OOD、So-Fake-Set） | —（无私有） |

**关键观察**：Document 域 93% 以上、DeepFake 域 72% 是私有数据，AIGC 域约 35% 是私有。**公开可复现的部分大约只占三分之一。**

### 3.3 闭环训练与难例挖掘（Figure 3）

流程：

1. 在数据池上训练 DeFakerOne；
2. 收集失败案例，送入 **agent 辅助的精修模块**；
3. agent 分析这些坏例，识别缺失的伪造模式、困难操纵类型与域；
4. agent 选择合适的生成/编辑模型，**自动调用模型或 API、选择提示词与源图，批量合成针对性样本**；
5. 新数据加入训练池，进入下一轮优化。

各域的执行方式不同：

- **AIGC**：维护一个商用 API 与开源生成/编辑模型的集合，新模型出现时 agent 自动调用并批量合成；
- **Document**：先匹配目标文档类别，再施加文本替换、印章修改、版面编辑、局部区域操纵等操作；团队另建有覆盖 4,000+ 类真实文档、凭证、合同、发票、证书的私有文档池；
- **DeepFake**：维护大规模开源真实人脸池（涵盖多样身份、头部姿态、表情、光照、遮挡、背景、分辨率），agent 调用人脸操纵模型合成样本，并显式考虑姿态、表情、身份与拍摄环境的变化；
- **Nature**：agent 用预分割区域与操作专用提示词生成局部操纵，包括拼接、复制粘贴、物体移除、inpainting、生成式局部编辑，并附带对应掩膜。

> 这是全文最有工业味的部分：把「人工造数据」变成「agent 驱动、以失败案例为信号的造数据」，思路与 active learning / hard example mining 一脉相承，但落到了生成侧。

### 3.4 GPT-Image-2-Bench

为评估对最新闭源基础生成模型的鲁棒性而自建的**小规模压力测试集**：

- **71 张**测试样本，六个类别均衡分布：Document 20%、DeepFake 20%、自然场景 20%、通用 AIGC 20%、海报 10%、社交媒体 App 风格内容 10%；
- 构造方式：用 **gemini-3-flash-preview** 作 VLM/LLM 骨干，对 OpenMMsec 的样本生成描述，再用 **GPT-Image-2** 重生成对应图像（Document / DeepFake / Nature 三类）；AIGC 类从 DiffusionDB（约 200 万真实用户提示词）采样提示词；海报类由 LLM 合成海报主题；社交媒体类覆盖小红书、抖音、Twitter、微信、微博、Instagram、QQ、Telegram 等平台风格。

设计意图：前四类保证主流生成内容的覆盖，后两类引入**文本密集、版式复杂、风格驱动**的更困难场景。

> 规模只有 71 张，1 张图约等于 1.4 个百分点。引用其准确率时必须说明这是小规模压力测试。

---

## 4. 方法：DeFakerOne

### 4.1 总体架构（Figure 7）

两个级联组件：

1. **MLLM 感知与检测模块**（InternVL2）：感知图像并做粗粒度检测，输出二值真实性判断 + 用于下游细粒度分割的 **segmentation token**；
2. **SAM2 分割模块**（SA2VA 式架构）：利用上述取证信息做像素级分析，输出定位掩膜。

整体是「MLLM 出判据 + 出提示，SAM2 出精细掩膜」的分工。作者强调选择 InternVL2 的理由：**反伪造任务优先关注局部低层伪影，而这些取证指标与图像语义内容没有确定性关联**——这正是下一步要改造检测范式的动机。

### 4.2 从二分类到动态 VQA

本文的一个关键设计：**不把 MLLM 的输出简单耦合到一个二值真假标签上，而是设计一组反伪造专用的动态 VQA 模板**。做法是：同一个问题配上**取决于真值标签的正向或负向答案**，从而把检测训练纳入标准 VQA 训练范式。

Table 1 给出 10 组问答模板，例如：

| 问题 | 正向答案 | 负向答案 |
|---|---|---|
| Are there any signs of tampering in this image? | Yes, signs of human tampering can be observed in the image. | No, no signs of human tampering were found in the image. |
| Does this image show evidence of being manually altered? | Yeah, we detected traces of artificial modification in the image. | Not, this image shows no evidence of any artificial modification. |
| Canyou tell if this image has been tampered with? | True, this image exhibits signs of artificial manipulation. | Never, We did not find any signs of tampering in the image. |
| Are there any indications of human editing in this image? | Sure, traces of human tampering can be identified in the image. | None, no signs of artificial alteration were detected in this image. |
| Does this image show signs of having been processed? | Sure, the image displays evidence of human modification. | Never, no signs of human tampering were detected in the image. |
| Is there any evidence of human manipulation in this image? | Yeah, we observed traces of artificial tampering in the image. | Not, this image does not appear to have been tampered with by humans. |
| Can you identify whether this image has been edited? | Yes, this image shows signs of being manually modified. | Never, there is no evidence indicating this image has been altered by humans. |
| Has this image been manually modified? | True, tampering traces can be identified in the image. | None, no traces of artificial modification exist in the image. |
| Are there any signs of editing in this image? | Yes, artificial alterations were detected in the image. | No, there are no signs of tampering in this image. |
| Does this image appear to have been tampered with? | Sure, the image shows visible signs of digital tampering. | None, we did not find any evidence of human modification in the image. |

设计动机（作者原话）：「增强标注多样性、**降低对图像语义的依赖**」。同时，遵循标准 VQA 训练范式有两个好处——既引导模型做检测，又**缓解通用问答能力的退化**。

> 汇报要点：这 10 个模板的答案词正好对应推理阶段受限词表的 8 个词（Yes/Yeah/True/Sure/No/Not/Never/None），训练与推理是配套设计的。

### 4.3 分割模块

- LLM 以多任务方式工作：输出检测 token 做全局分类，同时生成一组专用的 **segmentation token**；
- 这些 token 不只是抽象表征，而是**编码了潜在操纵的位置与性质的高层语义伪影**，作为分割头的**动态提示**；
- SAM2 编码器提取多尺度层次特征，保留粗粒度语义上下文与细粒度纹理细节；
- 融合机制的核心在 SAM2 decoder：LLM 产生的分割 token 通过 **cross-attention** 与视觉特征交互，引导 decoder 聚焦检测阶段识别出的可疑区域；
- decoder 输出高分辨率掩膜，用分割损失优化，以应对正负样本比例严重失衡的问题。

### 4.4 训练目标

给定输入图像 $I$ 与文本查询 $T$，MLLM 先输出检测 token $T_{Det}$ 与分割 token $T_{Seg}$，随后图像特征与 $T_{Seg}$ 一起送入分割层 $S$ 生成掩膜 $M$。总体 SFT 目标是文本项与分割项的加权和：

$$
\mathcal{L}_{SFT}=\lambda_{txt}\mathcal{L}_{txt}+\lambda_{seg}\mathcal{L}_{seg}
$$

其中文本项是标准的自回归交叉熵：

$$
\mathcal{L}_{txt}=-\sum_{i=1}^{N}\log P(x_i \mid X_{<i},I)
$$

分割项由 BCE 与 Dice 组成：

$$
\mathcal{L}_{seg}=\mathcal{L}_{BCE}(M,\hat{M})+\mathcal{L}_{Dice}(M,\hat{M})
$$

$\hat{M}$ 为真值掩膜，$\lambda_{txt}$ 与 $\lambda_{seg}$ 用于平衡文本生成与视觉分割两个目标。

### 4.5 三阶段训练

| 阶段 | 目的 | 关键设置 |
|---|---|---|
| **Stage-1 范式验证** | 域收敛验证 | 从 InternVL2 checkpoint 初始化，在 2M 四域图文对上做**全参 SFT**。目的是证明模型能同时获得多样伪造模式的判别特征、**且不出现任务目标间的灾难性干扰**。数据量较小 + 全参更新使模型快速适配取证范式。 |
| **Stage-2 能力扩展与规模化** | 大规模多任务训练 | 在 12.5M 样本上继续**全参 SFT**；AdamW，**1 个 epoch**，峰值学习率 $1\times10^{-5}$，warmup 比例 0.05，每卡 batch size 2，线性衰减 + 余弦退火；采用**平衡域采样**缓解四类数据不均衡；语料包含对抗扰动伪造与跨域复合操纵等困难边界样本。 |
| **Stage-3 多任务联合精修** | 解耦精修 + 分割对齐 | 语言侧：对 LLM 层施加 **LoRA（rank $r=128$，缩放因子 $\alpha=16$）**，视觉编码器与 connector **冻结**，学习率降至 $1\times10^{-6}$，防止过拟合并保留已获得的取证推理模式；分割侧：接入 **SAM 模块**，仅在 **Document + Nature 的 340K 图文掩膜对**上训练，SAM 主干全参微调，AdamW，峰值学习率 $1\times10^{-5}$，warmup 0.05，batch size 2，由域特定文本提示引导。 |

设计的逻辑：**解耦**让 MLLM 保留统一检测能力而不引起参数爆炸，同时让 SAM 模块专注结构复杂、像素边界对证据效力至关重要的伪造类型。

### 4.6 推理

**① 受限词表的篡改检测**。不直接依赖自由生成的回答，而是分析**第一个 token 在受限词表上的概率分布**：

$$
\mathcal{V}_{det}=\{\mathrm{Yes},\mathrm{Yeah},\mathrm{True},\mathrm{Sure},\mathrm{No},\mathrm{Not},\mathrm{Never},\mathrm{None}\}
$$

$$
p(v\mid I,T)=\frac{\exp(z_v)}{\sum_{u\in\mathcal{V}_{det}}\exp(z_u)},\qquad v\in\mathcal{V}_{det}
$$

其中 $z_v$ 是 token $v$ 的首 token logit。然后聚合正负回答 token 的概率：

$$
S_{tamper}=\sum_{v\in\{\mathrm{Yes},\mathrm{Yeah},\mathrm{True},\mathrm{Sure}\}} p(v\mid I,T)
$$

$$
S_{real}=\sum_{v\in\{\mathrm{No},\mathrm{Not},\mathrm{Never},\mathrm{None}\}} p(v\mid I,T)
$$

由于 $S_{tamper}+S_{real}=1$，最终判决等价于对 $S_{tamper}$ 用**固定 0.5 阈值**。好处是**免去任务特定阈值调优，并提供跨 FIDL 域统一的检测接口**。

**② SAM2 定位**。直接解码 MLLM 生成的分割 token：

$$
M=D_{SAM2}(F_I,T_{Seg})
$$

其中 $F_I$ 是输入图像的视觉特征表示，$D_{SAM2}$ 是 SAM2 解码器。分工是：MLLM 提供图像级真实性判断 + 高层定位引导，SAM2 做细粒度掩膜解码。

---

## 5. 实验结果

### 5.1 四域检测主结果（Table 2，24 个 benchmark）

四域平均汇总（括号内为最强基线）：

| 域 | 指标 | DeFakerOne | 最强基线 |
|---|---|---|---|
| DeepFake | AUC | **95.8** | CDFA 87.9 / Effort 88.2 |
| AIGC | ACC | **87.5** | Ivy-xDetector 79.9 / FakeVLM 78.5 |
| Document | ACC | **87.4** | DTD 61.6 / DRCT 62.4 |
| Nature | AUC | **86.7** | Mesorch 74.1 / TruFor 71.5 |

部分亮点与对比：

- **DeepFake**：CDFv1 / CDFv2 达 **99.9**，FF-DF 99.8。而 Veritas 在整个 DeepFake 域平均只有 **13.6**（ScaleDF 例外为 81.0），说明它在域外几乎失效。
- **AIGC**：GenImage 99.7、DiffusionForensics 99.5；Chameleon **84.7**，显著高于 Ivy-xDetector 的 73.2。
- **Document**：DocTamper_SCD 99.8、TestingSet 99.6、TextForensicsReasoning 99.6；FakeVLM 在 Document 域平均仅 **28.5**（DocTamper_FCD 只有 1.9），Ivy-xDetector 36.2——**缺文档训练数据直接导致崩盘**。
- **Nature**：AutoSplice 99.9、OpenSDI 98.6；但在 CASIAv1（96.3）以外的传统小数据集上优势相对温和（COVERAGE 75.9）。

### 5.2 跨域泛化（Table 3，OpenMMsec）

| 方法 | DeepFake | AIGC | IMDL | Doc | 平均 |
|---|---|---|---|---|---|
| SegFormer | 80.7 | 85.9 | 81.7 | 73.4 | 80.4 |
| Swin | 79.0 | 85.4 | 82.9 | 72.0 | 79.8 |
| Effort | 85.0 | 81.9 | 83.7 | 69.6 | 80.1 |
| FakeVLM | 75.8 | 86.6 | 66.2 | **14.0** | 60.7 |
| Veritas | 83.4 | 72.8 | 66.8 | 32.1 | 63.8 |
| **DeFakerOne** | **89.5** | **96.4** | **91.1** | **90.1** | **91.8** |

论文的解读很直白：**小视觉模型反而超过那些没在 OpenMMsec 上训练过的 MLLM**，说明后者的跨域泛化有限；FakeVLM 与 Ivy-xDetector 在 Doc 域低于 20%，根源是训练数据里缺 Document 类别。DeFakerOne 以 91.8% 的平均准确率领先。

### 5.3 定位精度（Table 4，像素级二值 F1）

| Document benchmark | DTD | CAFTB | TIFDM | DeFakerOne |
|---|---|---|---|---|
| 平均 | 67.4 | 52.1 | 41.4 | **78.7** |

| Nature benchmark | MVSS-Net | TruFor | Mesorch | DeFakerOne |
|---|---|---|---|---|
| 平均 | 42.5 | 50.3 | 52.0 | **67.4** |

在 Document 的 7 个 benchmark 中 5 个第一；在 Nature 的 6 个中 4 个第一（COVERAGE、NIST16、CocoGlide、AutoSplice），CASIAv1 与 Columbia 属竞争力水平。

注意 Nature 的 NIST16：DeFakerOne 65.6，而 MVSS-Net 仅 29.4、TruFor 34.8、Mesorch 39.2——**这项提升幅度最大**，与后文「分割监督对困难局部操纵最有效」的结论一致。

### 5.4 鲁棒性（Table 5，OpenMMsec 平均）

| 扰动 | FFDN | Mesorch | Effort | ForensicsAdapter | ForensicMOE | FakeShield | **DeFakerOne** |
|---|---|---|---|---|---|---|---|
| Gaussian Blur | 47.63 | 49.45 | 56.71 | 55.76 | 54.11 | 64.63 | **79.46** |
| Brightness | 44.78 | 51.06 | 57.53 | 56.10 | 54.82 | 63.48 | **70.60** |
| Contrast | 43.65 | 51.12 | 58.27 | 56.22 | 55.53 | 63.86 | **76.26** |
| JPEG Compression | 39.93 | 52.00 | 58.73 | 56.78 | 51.67 | 62.77 | **76.16** |
| Noise | 48.03 | 48.18 | 54.62 | 52.77 | 53.62 | 63.54 | **65.32** |
| Resize | 44.19 | 50.54 | 58.69 | 57.02 | 53.47 | 51.30 | **69.23** |
| Saturation | 40.69 | 51.69 | 58.43 | 57.30 | 55.77 | 64.30 | **81.85** |

作者的解释：DeFakerOne 学到的是更泛化、对扰动不变的表示，而不是脆弱的低层线索；在 resize 与 JPEG 上尤其明显，在噪声上相对较弱。

> 我在附录 Table 12 中发现一处异常：Brightness 等级 1.5 的准确率是 **50.33**，而相邻等级 1.0 / 2.0 分别是 84.59 / 70.36，跳变突兀；正是这个值把 Brightness 行的平均从约 80 拉到 70.60。可能是评测或记录问题，引用时留意。

### 5.5 推理稳定性（Table 7）

| 随机种子 | ACC | F1 | | 温度 | ACC | F1 |
|---|---|---|---|---|---|---|
| 42 | 91.8 | 93.4 | | 0.1 | 91.8 | 93.4 |
| 1024 | 91.7 | 93.5 | | 0.5 | 91.8 | 92.5 |
| 8192 | 91.8 | 93.5 | | 0.9 | 91.7 | 93.5 |

三个种子、温度 0.1–0.9 范围内波动仅 0.1 个百分点。作者认为这说明模型学到的是稳健的取证决策边界，而非脆弱的虚假相关——对需要**证据可复现性**的取证场景是重要前提。

### 5.6 GPT-Image-2-Bench（Figure 8）

| 方法 | ACC (%) |
|---|---|
| **DefakerOne-2B（本文）** | **95.8** |
| FakeVLM | 71.8 |
| Veritas | 71.8 |
| ForensicMOE | 63.4 |
| Ivy-xDetector | 56.3 |
| FakeShield | 50.7 |
| Mirror | 46.5 |
| DRCT | 45.1 |
| DDA | 43.7 |
| AIDE | 19.7 |

（正文只点名了 DRCT、DDA、Mirror、FakeShield 四者低于 51%，实际上 AIDE 的 19.7% 是全表最低。）

作者的解释：GPT-Image-2 语义连贯性更强、局部纹理更干净、低层合成伪影更少，使传统依赖伪影的检测器失效。要检测这类图，**不仅需要低层伪影识别，还需要对语义一致性、版式合理性与跨区域视觉证据的高层取证推理**。

### 5.7 消融：域与任务的必要性（Table 6）

| 模型 | Doc | IMDL | Deepfake | AIGC | 平均 |
|---|---|---|---|---|---|
| DeFakerOne（Stage-3 多任务） | 92.5 | 89.7 | 89.0 | 96.8 | **92.0** |
| DeFakerOne（Stage-2 数据规模化） | 90.1 | 91.1 | 89.5 | 96.4 | 91.8 |
| DeFakerOne（Stage-1 多域） | 89.7 | 54.5 | 77.4 | 85.9 | 76.8 |
| Nature 单域专用 | 66.6 | 74.6 | 64.3 | 74.5 | 70.0 |
| AIGC 单域专用 | 13.9 | 53.2 | 28.3 | 92.9 | 47.1 |
| Deepfake 单域专用 | 13.8 | 47.6 | 88.7 | 45.8 | 49.0 |
| Doc 单域专用 | 89.5 | 55.5 | 29.2 | 44.3 | 54.6 |

**这个消融是全文最有说服力的一张表**：单域专用模型在目标域上表现不错（AIGC 专用 92.9、Deepfake 专用 88.7、Doc 专用 89.5），但一旦跨域就崩塌（AIGC 专用在 Doc 上仅 13.9，Deepfake 专用在 Doc 上 13.8）。统一多域训练的平均值是 92.0，远高于任何单域专家的 70.0。

---

## 6. 五个数据规律（Result Analysis，本文的方法论内核）

### 6.1 数据规模化不等于性能提升（Figure 9）

极端单域扩增实验：DeepFake 域原本 2.336M 人脸伪造训练样本，额外加入约 **14M** 张 ScaleDF 样本，总量升到约 **16.336M**。

结果：开源 DeepFake benchmark 上的性能不但没有继续提升，反而**相对下降约 4.9%**。

结论：**即使是目标域自身，持续扩大单域数据规模也不必然带来更好的性能**。模型可能很快到达性能平台期，或因分布偏移、样本冗余、数据组成失衡而退化。FIDL 性能**不遵循简单的「数据越多越好」的 scaling law**；关键在**比例、质量与分布互补性**。

### 6.2 跨域转移与干扰由「操作级伪影」决定（Figure 10 + Table 8）

渐进式域专项数据扩增的结果是混合的转移—干扰行为：

- 加入 Nature 数据后，Nature 性能提升约 **16.5%**，AIGC 也随之提升，但 **Doc 与 DeepFake 下降**；
- 说明 Nature 数据不仅惠及目标域，还能**转移**到伪影模式兼容的非目标域。

这构成一个表面矛盾：加一个域有时惠及其他域，有时又压制其他域。作者指出这**无法用 Nature / AIGC / DeepFake / Document 这样的宏观域标签解释**，于是下沉到更细的粒度分析。

Table 8（在 Nature-only 基础上加入 AIGC + Doc 数据）：

| 设置 | Nature 平均 | CocoGlide | AutoSplice | OpenSDI |
|---|---|---|---|---|
| Gain over Nature-only | **-1.6%** | **+9.48%** | **+20.42%** | **+13.83%** |

即：虽然 Nature 域平均略微下降，但其中涉及 **AIGC 风格的局部操纵、生成式编辑或语义补全**的子集反而大幅提升。

结论：**跨域转移与干扰主要由操作级伪影的相似性决定**——生成式纹理偏置、语义补全痕迹、融合不一致、边界伪影、换脸痕迹、文本替换伪影——即便属于不同宏观域，也能互相受益；反之，不兼容的伪影模式会引入干扰。因此 **FIDL 数据不应只按域组织，还应按操纵方法组织。**

### 6.3 平衡的数据重构是统一 FIDL 的关键

在识别出操作级伪影导致的转移—干扰后，作者进一步问：如何在这种跨域交互下稳定统一训练？

做法是按「目标域增强 → 跨域干扰 → 补强弱化域 → 全局再平衡」的路径重构数据。在 Figure 10 的最后阶段，继续补充 Document 与 AIGC 数据后：

- Doc 平均性能恢复约 **+6.0%**；
- AIGC 平均性能提升约 **+5.9%**；
- DeepFake 保持稳定并略有提升；
- Nature 仅有约 **1.2%** 的小幅波动。

四域总体平均提升约 **9.6%**。作者强调：这个增益**不是来自总数据量的单调增长**，而是来自数据重构过程。最终 DeFakerOne 的数据组成让四个域维持**量级相当**的规模，从而在检测、定位与跨域泛化上同时保持性能。

**方法论要点：统一 FIDL 的关键不是让某个域主导，而是在平衡且操作感知的多域规模下重新组织数据。**

### 6.4 不同域需要不同的监督粒度（Table 9）

观察：DeepFake 与 AIGC 相对容易从图像级标签学到（DeepFake 的操纵集中在人脸区域，AIGC 图像常有更稳定的全局生成痕迹）；而 **Nature 与 Doc 的伪造更细粒度**——Nature 常涉及局部拼接、复制粘贴、物体移除或生成式局部编辑，Doc 可能只修改很小的文本区域、数字、印章或局部版面。此时关键取证线索是**局部且微弱的**，容易被全局语义表征淹没。

验证：引入分割监督，观察对 Nature 分类能力的增益。

| 监督 | Coverage | Columbia | NIST16 | CocoGlide | Autosplice | DSO-1 | 平均 |
|---|---|---|---|---|---|---|---|
| 增益（%） | +1.1 | +2.0 | **+10.7** | +2.9 | +0.4 | +2.5 | **+3.3** |

结论：**联合分类与分割训练不仅提升定位能力，也提升图像级分类性能**——像素级掩膜帮助模型聚焦操纵边界、局部残差与区域不一致。这解释了「细粒度攻击 vs 粗粒度防御」的不对称：只靠图像级真伪标签的防御模型抓不住这些微弱局部痕迹。

### 6.5 保持原分辨率伪影对统一 FIDL 至关重要（Table 10）

用 InternVL2-2B 作基线，对比更新的通用多模态骨干：

| 骨干 | DeepFake | Document | AIGC | Nature | 平均 |
|---|---|---|---|---|---|
| InternVL2-2B（基线） | – | – | – | – | – |
| InternVL3.5-2B | -0.6% | **-13.2%** | -1.2% | -2.0% | -4.3% |
| Qwen3-VL-2B | -0.4% | **-10.4%** | +5.0% | +1.5% | -1.1% |

更新的通用 VLM 在 Nature 或 AIGC 上可能带来增益，但在 **Document 上明显退化**。作者的归因：**视觉信息保留方式**不同。InternVL2 采用**动态高分辨率切块（tiling）**策略，更好地保留了视觉编码过程中的局部像素级细节；而更新的骨干更强调推理效率、视觉 token 预算控制与语义聚合，引入了更强的视觉压缩。这种压缩对一般视觉语言任务有益，但可能**稀释甚至抹掉 FIDL 中的弱取证伪影**——文本边缘变化、局部版面异常、边界不连续、纹理残差、压缩痕迹。

结论：**统一 FIDL 需要能保留高分辨率局部证据的视觉骨干，而不是简单依赖更新、更强的通用 VLM。**

---

## 7. 结论与未来工作

### 四条核心结论（作者自己总结）

1. **超越无约束规模化**：数据规模与性能不呈线性关系，有效模型取决于刻意的数据组成平衡；
2. **操作级伪影感知**：跨域转移主要由底层操纵机制的相似性决定，而非宏观域标签；未来应按操作足迹组织取证数据；
3. **多粒度监督**：全局标签对部分任务足够，但细粒度像素监督对「细粒度局部操纵 vs 粗粒度防御模型」的不对称必不可少；
4. **原分辨率伪影保留**：FIDL 骨干设计应优先保证高分辨率局部证据，而不是只看模型规模或通用 VLM 性能。

### 未来三个方向

1. **可扩展基础模型与数据工程**：从独立模型演进为支撑产学研各类应用的基础设施；关键在高效的数据管线，能大规模生成高保真、多样、有代表性的取证数据，并探索在「in-the-wild」环境下提升泛化的架构创新；
2. **Agent 范式注入专家知识**：核心张力在于 **outcome injection 与 process injection 的二分**——前者通用但受限于高质量专家标签难以获取，后者（如构造 CoT 数据）目前受内容同质化、高度依赖人工设计、知识迁移性差所限；作者认为未来在于把**真实、低歧义的真值数据**与 agent 工具结合，从静态检测走向基于证据的推理；
3. **多模态与物理—数字统一取证**：跨模态集成（扩展到视频与音频认证）；物理—数字联合（把数字伪造检测与物理反欺骗如面具、重放、打印攻击连通，并引入设备硬件、传感器噪声签名、环境因素等物理上下文元数据）。

---

## 8. 我的评价

### 站得住的部分

1. **Table 6 的域消融是全文最有说服力的证据**。单域专家模型在跨域时崩塌到 13.8–54.6 的平均水平，而统一多域训练达 92.0。这直接证明了「统一 FIDL」不是噱头而是必需，也是本文最重要的科学结论。
2. **数据规律的实验设计有方法论价值**。5 个规律里有 4 个是**反直觉的负面结果**（单域扩了 14M 反而 -4.9%；加 Nature 数据让 Doc/DeepFake 下降；更新的骨干在 Doc 上退化 13.2%），而负面结果往往比正面结果更有信息量。这种「把失败做成方法论」的写法很值得学习。
3. **「操作级伪影决定转移」这一结论可操作**。它给出了具体的行动指南——数据应按**操纵方法**组织而非仅按域组织——而不只是停留在「数据很重要」这种空话。
4. **推理设计兼顾了取证场景的实际需求**。受限词表 + 固定 0.5 阈值，避免任务特定调参，提供了一个跨域统一的判定接口；三类 seeds × 五档温度的稳定性测试则回应了取证结果需要可复现的合规诉求。
5. **工程完整度高**。数据构造 → 训练三阶段 → 推理 → 跨域/鲁棒性/定位的全套评测，加上 24 个检测 benchmark + 9 个定位 benchmark，覆盖度远超一般学术论文。

### 需要打折扣或存疑的部分

1. **可复现性是这个工作最大的软肋（必须主动说明）**。Document 域 2.5M 中 93% 以上是私有数据，DeepFake 域约 72% 私有，AIGC 域约 35% 私有。论文只给了 GitHub 链接，第三方很难复现 Table 2 的结果。更重要的是，**Table 6 的域消融依赖这套私有数据**，因此「统一多域训练优于单域专家」这一核心结论在公开数据上能否复现，论文并未验证。
2. **Table 6 的对比范式存在公平性问题**。单域专用模型是在该域数据上训练的，而 DeFakerOne 用的是全部四域数据（规模也更大）。因此「统一 > 单域专家」的结论里，有多少来自**多域数据**、有多少来自**数据总量更大**，两者被混淆了。一个更干净的对照应该是「四域数据按总量与单域专家一致」的设置。论文没有做这个控制。
3. **缺少与最强 MLLM 检测器的同规模对比**。"单域专家 vs 统一模型"是本文的论点，但表格里的 MLLM 基线（FakeVLM、Veritas、Ivy-xDetector）**并未在这套 12.5M 数据上训练过**，所以在 Table 2/3 中它们输给 DeFakerOne，无法区分是「架构更好」还是「数据更多更好」。作者在附录 A.2.3 也承认这些 MLLM 方法「更偏向推理链生成或描述性解释，而非核心的 FIDL 目标」——这既是批评也是免责，读者需自行判断公平性。
4. **GPT-Image-2-Bench 规模过小**。71 张、六个类别，每类 7–14 张。1 张图约 1.4 个百分点，且样本由 VLM 描述 OpenMMsec 后由 GPT-Image-2 重生成，与真实世界的生成分布有差距。**95.8% 这个数字应作为「小规模压力测试上的领先」而非稳固的绝对性能**来引用。
5. **鲁棒性表存在未解释的异常**。Table 12 中 Brightness 等级 1.5 的 50.33 与相邻等级的 84.59 / 70.36 明显不连贯，且它把该行平均值拉低了约 10 个点。论文未作说明。
6. **「模型 vs 数据」的归因不够彻底**。本文反复强调「数据组成比数据规模重要」，但所有的数据实验都建立在**同一个模型与同一套训练流程**上，也没有给出「同等算力下不同数据配比」的对照。作为方法论文章，如果能给出「相同 token 预算 / 相同训练步数下不同域配比」的曲线，结论会更硬。
7. **论文体例偏松散**。21 页正文 + 附录的规模更像技术报告；「39 forgery detection benchmarks and 9 localization benchmarks」与正文声称的「40 benchmarks」表述不一致；文中对 contributor 的列出方式（附录 C）也说明这是团队工程总结而非严格的学术论文，引用时建议标注为预印本/技术报告。

### 一句话

**如果只读一篇来理解「工业界怎么做统一 FIDL」，就是这一篇**——它的价值在数据方法论（Table 6 / Figure 9 / Figure 10 / Table 9 / Table 10）与完整评测，而不在模型结构；同时它的可复现性与对比公平性是必须向听众交代的前提。

---

## 9. 汇报时的讲稿骨架（4 分钟）

1. **动机**（40s）：Figure 2——过去四个域各有各的伪影假设，现在基础生成模型把域边界抹平了，域专用方法失效。举 GPT-Image-2 为例：纹理更干净、语义更一致，传统伪影检测器直接掉到 50% 以下。
2. **数据**（70s）：12.5M、四域构成、三阶段（2M 验证 → 12.5M 扩展 → LoRA+SAM 精修）。特别讲**闭环难例挖掘**：模型失败 → agent 分析 → 反向工程操纵链 → 生成针对性数据 → 重训。同时坦白：私有数据占约三分之二，复现困难。
3. **模型**（40s）：InternVL2 + SAM2；两个关键设计——(a) 10 组动态 VQA 模板，把二分类变成 VQA，降低对语义的依赖；(b) 受限词表 8 个词 + 固定 0.5 阈值，跨域统一、免调参。
4. **结果**（40s）：四域平均全面领先（DeepFake 95.8 AUC / AIGC 87.5 ACC / Doc 87.4 ACC / Nature 86.7 AUC）；OpenMMsec 91.8；GPT-Image-2-Bench 95.8（多数基线 <51）。
5. **数据规律**（60s，本文精华）：放 Figure 9 和 Figure 10。三句话——单域加 14M 数据反而 -4.9%；跨域转移由**操作级伪影**决定而不是域标签；最终靠平衡重构让四域平均 +9.6%。再加一个反直觉点：Table 10 显示**更新的通用 VLM 骨干在 Document 上退化 10–13%**，因为视觉压缩抹掉了弱伪影。
6. **收束**（20s）：**FIDL 不缺模型，缺的是数据配比方法论。**
