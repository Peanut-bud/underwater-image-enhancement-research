这几篇计算机视觉（CV）领域的顶级顶会/顶刊论文，学术界都有开源的惯例。你完全不需要花钱去买数据库权限，所有的论文原稿（Preprint 版本）以及大多数的官方代码都可以免费合法地获取。

我为你整理了这几篇核心论文的免费合法下载链接（主要是 arXiv 预印本库和 CVF 计算机视觉基金会开源库）以及它们的官方开源代码地址。你可以直接点击下载 PDF：

### 1. 支撑“物理模型合成数据”的经典论文

- **AOD-Net (ICCV 2017)**
  - *全称:* AOD-Net: All-in-One Dehazing Network
  - *地位:* 将物理公式 $K$ 参数化的开山之作。
  - *免费 PDF 下载 (CVF官方):* [点击下载 AOD-Net PDF](https://openaccess.thecvf.com/content_ICCV_2017/papers/Li_AOD-Net_All-In-One_Dehazing_ICCV_2017_paper.pdf)
- **UWCNN (Pattern Recognition 2020)**
  - *全称:* Underwater Image Enhancement via a Global–Local Networks and Objective-Predicting Model (通常提及其利用合成水下数据集的方法)
  - *替代/关联开源推荐 (更知名)*: Anwar 团队的深入综述与合成方法 *Diving Deeper into Underwater Image Enhancement: A Survey* (含合成数据集方法)。
  - *免费 PDF 下载 (arXiv):* [点击查看/下载预印本](https://arxiv.org/abs/1903.09766)

### 2. 支撑“分解/解耦架构”的核心论文

- **UColor (TIP 2021)**
  - *全称:* UColor: An Underwater Image Enhancement Framework Based on Color Space Feature Fusion
  - *地位:* 结合了透射率图与颜色空间的特征融合，极度契合你的问题。
  - *免费 PDF 下载 (arXiv):* [点击下载 UColor PDF](https://arxiv.org/abs/2101.00973)
  - *官方代码:* [UColor GitHub](https://github.com/Li-Chongyi/Ucolor)
- **Retinex-Net (BMVC 2018)**
  - *全称:* Deep Retinex Decomposition for Low-Light Enhancement
  - *地位:* “光照与反射率解耦”的经典网络。
  - *免费 PDF 下载 (arXiv):* [点击下载 Retinex-Net PDF](https://arxiv.org/abs/1808.04560)
  - *官方代码:* [Retinex-Net GitHub](https://github.com/weichen582/RetinexNet)

### 3. 支撑“主干/精炼网络”的 SOTA 论文

- **NAFNet (ECCV 2022)**
  - *全称:* Simple Baselines for Image Restoration
  - *地位:* 极简、极快、极省显存的图像恢复天花板。
  - *免费 PDF 下载 (arXiv):* [点击下载 NAFNet PDF](https://arxiv.org/abs/2204.04676)
  - *官方代码:* [NAFNet GitHub](https://github.com/megvii-research/NAFNet)
- **Restormer (CVPR 2022)**
  - *全称:* Restormer: Efficient Transformer for High-Resolution Image Restoration
  - *地位:* 解决极端高光和复杂纹理丢失的最强 Transformer 架构。
  - *免费 PDF 下载 (arXiv):* [点击下载 Restormer PDF](https://arxiv.org/abs/2111.09881)
  - *官方代码:* [Restormer GitHub](https://github.com/swz30/Restormer)

### 4. 支撑“无参考/零样本微调”的论文

- **Zero-DCE (CVPR 2020)**
  - *全称:* Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
  - *地位:* 提出了“颜色恒常性损失”、“空间平滑损失”等多种不需要真实清晰图的无监督损失函数。
  - *免费 PDF 下载 (CVF官方):* [点击下载 Zero-DCE PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
  - *官方代码:* [Zero-DCE GitHub](https://github.com/Li-Chongyi/Zero-DCE)

------

**💡 阅读建议：**

如果你打算这周开始动手，我强烈建议你**优先阅读 NAFNet 和 Zero-DCE** 这两篇论文的 **Methodology (方法/网络架构)** 章节。

- 读 NAFNet 是为了知道怎么在 PyTorch 里把那个最牛的 Block 抄过来（替换掉我之前写的简单卷积）；
- 读 Zero-DCE 是为了学习它那几个绝妙的无监督 Loss 函数公式，这直接决定了你手里那批真实图片能不能被完美修复。
