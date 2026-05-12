# 数据目录

`data/` 是本工程唯一的标准数据入口，`数据/` 仍然是原始资料库，保持只读，不改原始结构。

当前固定结构：

```text
data/
├─ raw_field/
│  ├─ images/
│  ├─ videos/
│  └─ manifests/
├─ clean_source/
│  ├─ images/
│  └─ manifests/
├─ synthetic/
│  ├─ train/
│  │  ├─ input/
│  │  ├─ target/
│  │  ├─ transmission/
│  │  ├─ airlight/
│  │  └─ metadata/
│  ├─ val/
│  └─ test/
├─ real_unsup/
│  ├─ train/
│  ├─ val/
│  └─ masks/
└─ splits/
```

语义约定：

- `raw_field/`：从 [数据](/D:/科研/图像增强/数据) 中整理出的真实现场原图入口，只记录索引和工程所需子集。
- `clean_source/`：用于阶段 1 物理退化合成的清晰参考图。
- `synthetic/`：阶段 1 产出的正式监督数据，包含四元组和元数据。
- `real_unsup/`：阶段 4 使用的真实无配对图像。
- `splits/`：统一记录 train / val / test 样本清单。

当前仓库内放置了一小批 smoke 级样例，用途仅限于：

- 验证四元组 dataset 读取协议
- 验证监督训练与微调训练入口
- 验证推理与评估入口

这些样例不是正式论文训练集，也不代表最终数据分布。
