# 图像增强研究工程

本项目面向**人工光照水下场景图像增强**，目标是把“论文思路 + 研究代码 + 训练/推理流程”整理成一个统一、可扩展的研究工程。

当前工程不是空骨架了，而是已经具备：

- 单图与批量推理闭环
- 最小训练闭环
- 可训练主干模型
- 第一版高光感知分支
- 基础测试与 smoke test

---

# 项目目标

目标输入输出形式：

```text
输入：单张或多张水下原始图像
输出：增强后的图像
```

重点处理的问题包括：

- 黄绿色偏色
- 浑浊发灰、低对比度
- 纹理模糊
- 左亮右暗、非均匀光照
- 强反光
- 局部过曝
- 暗区细节不足

项目最终希望同时支持：

- 模型训练
- 单图推理
- 批量推理
- 定量评估
- 消融实验
- 论文结果整理

---

# 当前实现状态

## 已完成

- 已搭建统一工程结构
- 已建立 GitHub 仓库
- 已完成最小推理闭环
- 已完成最小训练闭环
- 已实现基础可训练主干
- 已实现第一版高光感知分支
- 已整理训练 smoke test 数据子集
- 已补齐基础测试

## 当前模型主线

当前核心模型已经包含：

```text
Input
-> Shallow Feature Extraction
-> Highlight-Aware Branch
-> Frequency Enhancement
-> Differential Detail Enhancement
-> Reconstruction
-> Output
```

当前更接近“第一阶段可运行研究系统”，还不是最终论文全模型。

## 还未完成

- 非均匀光照分支
- 混合曝光分支
- 独立噪声抑制分支
- 注意力融合模块
- 更完整的损失函数体系
- 正式监督训练数据方案
- 独立评估模块

---

# 环境说明

当前推荐环境：

- `D:\Anaconda\envs\DL2`

重要约束：

- `DL` 为强化学习环境，不作为本项目环境使用
- `DL2` 未经确认不做删除、重建或大范围替换
- 如需新增环境或删除环境，必须先确认

当前配置文件中记录的也是 `DL2`：

- [configs/infer_base.yaml](/D:/科研/图像增强/configs/infer_base.yaml)
- [configs/train_base.yaml](/D:/科研/图像增强/configs/train_base.yaml)

---

# 快速开始

## 1. 运行推理

默认会读取 [input_images](/D:/科研/图像增强/input_images) 中的样例图，输出到 [outputs/infer](/D:/科研/图像增强/outputs/infer)。

```powershell
conda run -n DL2 python -m src.infer --config ./configs/infer_base.yaml
```

当前推理入口：

- [src/infer.py](/D:/科研/图像增强/src/infer.py)
- [src/inference/pipeline.py](/D:/科研/图像增强/src/inference/pipeline.py)

## 2. 运行最小训练闭环

当前训练配置会读取 [data](/D:/科研/图像增强/data) 下整理好的 smoke test 子集，跑 1 个 epoch，并把权重保存到 [checkpoints](/D:/科研/图像增强/checkpoints)。

```powershell
conda run -n DL2 python -m src.train --config ./configs/train_base.yaml
```

当前训练入口：

- [src/train.py](/D:/科研/图像增强/src/train.py)
- [src/trainers/basic_trainer.py](/D:/科研/图像增强/src/trainers/basic_trainer.py)

## 3. 运行测试

```powershell
conda run -n DL2 python -m unittest tests.test_stage1_smoke
conda run -n DL2 python -m unittest tests.test_paired_dataset
conda run -n DL2 python -m unittest tests.test_trainable_enhancer
conda run -n DL2 python -m unittest tests.test_basic_losses
conda run -n DL2 python -m unittest tests.test_training_smoke
```

---

# 目录结构

```text
图像增强/
├─ configs/          训练、推理、评估配置
├─ data/             工程标准数据入口与 smoke test 子集
├─ input_images/     推理样例输入
├─ outputs/          推理结果与实验产物
├─ checkpoints/      训练权重
├─ src/              核心源码
├─ tests/            单元测试与 smoke test
├─ 思路/             方案文档、模型流程、论文代码映射
├─ 数据/             原始资料库（只读，不改原结构）
└─ 图像增强论文/      论文资料归档
```

补充说明：

- [数据](/D:/科研/图像增强/数据) 是原始资料库，后续继续保持只读，不改变原本内容结构。
- [思路/项目实施规划.md](/D:/科研/图像增强/思路/项目实施规划.md) 是总规划文档。
- [思路/模型流程.md](/D:/科研/图像增强/思路/模型流程.md) 是模型结构路线说明。
- [思路/代码.md](/D:/科研/图像增强/思路/代码.md) 是论文代码与工程模块映射手册。

---

# 关键源码说明

## 数据与预处理

- [src/datasets/paired_dataset.py](/D:/科研/图像增强/src/datasets/paired_dataset.py)
  - 读取 `data/train|val/input,target` 的配对数据

- [src/preprocessing/image_ops.py](/D:/科研/图像增强/src/preprocessing/image_ops.py)
  - 推理侧图像读写、归一化、尺寸恢复

- [src/preprocessing/transforms.py](/D:/科研/图像增强/src/preprocessing/transforms.py)
  - 训练侧 resize、crop、tensor 化

## 模型

- [src/models/minimal_enhancer.py](/D:/科研/图像增强/src/models/minimal_enhancer.py)
  - 最小推理闭环模型

- [src/models/trainable_enhancer.py](/D:/科研/图像增强/src/models/trainable_enhancer.py)
  - 当前核心可训练模型
  - 包含主干增强与高光感知第一版

## 损失

- [src/losses/basic_losses.py](/D:/科研/图像增强/src/losses/basic_losses.py)
  - `L1 + SSIM + Edge Loss`

## 训练与推理

- [src/train.py](/D:/科研/图像增强/src/train.py)
- [src/trainers/basic_trainer.py](/D:/科研/图像增强/src/trainers/basic_trainer.py)
- [src/infer.py](/D:/科研/图像增强/src/infer.py)
- [src/inference/pipeline.py](/D:/科研/图像增强/src/inference/pipeline.py)

---

# 当前数据说明

当前 [data](/D:/科研/图像增强/data) 下的训练数据主要用于**训练链路 smoke test**，不是最终正式训练集。

当前状态：

- 已从原始资料库中复制少量代表图构成最小子集
- 当前 `target` 采用“同图对同图”方式，仅用于验证训练链路
- 原始资料库 [数据](/D:/科研/图像增强/数据) 未被改动

详细说明见：

- [data/README.md](/D:/科研/图像增强/data/README.md)
- [data/DATASET_SMOKESET.md](/D:/科研/图像增强/data/DATASET_SMOKESET.md)

---

# 当前参考路线

目前工程参考路线已经基本明确：

- 主干增强：
  - 黄奕程等论文

- 高光与过曝：
  - HDR CNN
  - Over-exposure Correction
  - Specular Highlight Removal

- 非均匀光照：
  - UNIR-Net
  - Ning 等论文
  - NUIENet

- 混合曝光：
  - RECNet

- 注意力融合与对比学习：
  - 王悦等论文

- 基线与对比实验：
  - FUnIE-GAN
  - Water-Net
  - Ucolor
  - U-shape Transformer

详细映射见：

- [思路/代码.md](/D:/科研/图像增强/思路/代码.md)

---

# 下一步建议

按当前进度，下一步最自然的是继续把模型从“主干 + 高光感知”推进到更完整的多分支结构，优先建议：

1. 非均匀光照分支
2. 混合曝光分支
3. 注意力融合模块
4. 更正式的数据与评估方案

---

# 相关文档

- [思路/项目实施规划.md](/D:/科研/图像增强/思路/项目实施规划.md)
- [思路/模型流程.md](/D:/科研/图像增强/思路/模型流程.md)
- [思路/代码.md](/D:/科研/图像增强/思路/代码.md)
- [思路/解决方法.md](/D:/科研/图像增强/思路/解决方法.md)
- [思路/ai的解决方法.md](/D:/科研/图像增强/思路/ai的解决方法.md)
