# 图像增强研究工程

本项目面向人工光照水下场景，当前主线已经切换为“四阶段物理引导增强框架”：

1. 物理模型合成监督数据
2. 物理引导增强模型搭建
3. 基于四元组数据的监督预训练
4. 基于真实无配对图像的无监督微调

这次重构的目标不是立刻补全全部论文算法，而是先把工程骨架、配置边界、数据入口、训练入口和测试边界全部定死，后续实现可以直接按 [gemini流程.md](D:/科研/图像增强/思路/gemini流程.md:1) 继续填充。

## 当前框架状态

- 已清理旧的 `highlight/frequency/diff` 主线语义
- 已建立四阶段对应的目录、入口和测试骨架
- 已补齐合成监督数据与真实无监督数据的标准入口
- 已提供物理引导模型、监督训练、无监督微调的最小可运行占位实现
- 已重写配置体系，按 `data / model / train / infer / eval` 分层组织

## 项目目标

```text
输入：单张或多张水下原始图像
输出：增强后的图像
```

重点问题：

- 黄绿色偏色
- 发灰浑浊、低对比度
- 纹理边缘弱化
- 非均匀光照
- 强反光与局部过曝
- 暗区细节不足
- 真实域与合成域分布偏差

## 推荐环境

- `D:\Anaconda\envs\DL2`

当前框架层不强依赖某个环境细节，但默认文档和命令示例仍以 `DL2` 为准。

## 快速开始

阶段 1：检查合成数据构建配置

```powershell
conda run -n DL2 python -m src.prepare_synth --config ./configs/data/synth_build.yaml
```

阶段 3：运行监督训练 smoke test

```powershell
conda run -n DL2 python -m src.train_supervised --config ./configs/train/supervised_base.yaml
```

阶段 4：运行无监督微调 smoke test

```powershell
conda run -n DL2 python -m src.train_adapt --config ./configs/train/adapt_base.yaml
```

推理：

```powershell
conda run -n DL2 python -m src.infer --config ./configs/infer/infer_base.yaml
```

评估：

```powershell
conda run -n DL2 python -m src.eval --config ./configs/eval/paired_base.yaml
conda run -n DL2 python -m src.eval --config ./configs/eval/noref_base.yaml
```

测试：

```powershell
conda run -n DL2 python -m unittest discover -s tests -p "test_*.py"
```

## 目录结构

```text
图像增强/
├─ configs/          分层配置
├─ data/             工程标准数据入口
├─ input_images/     推理样例输入
├─ outputs/          推理结果与实验产物
├─ checkpoints/      训练权重
├─ src/              核心源码
├─ tests/            单元测试与 smoke test
├─ 思路/             方案文档与论文映射
├─ 数据/             原始资料库（只读）
└─ 图像增强论文/      论文资料归档
```

## 当前源码主线

- [src/prepare_synth.py](/D:/科研/图像增强/src/prepare_synth.py)
  - 阶段 1 入口，负责校验合成构建配置和目录
- [src/models/physical_guided_enhancer.py](/D:/科研/图像增强/src/models/physical_guided_enhancer.py)
  - 统一物理引导模型
- [src/train_supervised.py](/D:/科研/图像增强/src/train_supervised.py)
  - 阶段 3 监督训练入口
- [src/train_adapt.py](/D:/科研/图像增强/src/train_adapt.py)
  - 阶段 4 无监督微调入口
- [src/infer.py](/D:/科研/图像增强/src/infer.py)
  - 单图与批量推理入口
- [src/eval.py](/D:/科研/图像增强/src/eval.py)
  - paired / no-reference 两类评估入口

## 文档索引

- [data/README.md](/D:/科研/图像增强/data/README.md)
- [configs/README.md](/D:/科研/图像增强/configs/README.md)
- [思路/gemini流程.md](D:/科研/图像增强/思路/gemini流程.md)
- [思路/代码.md](D:/科研/图像增强/思路/代码.md)

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
