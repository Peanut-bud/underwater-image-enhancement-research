# 数据目录

本目录用于存放**规范化后的工程数据**，不替代现有的 `数据/` 原始资料目录。

建议后续使用如下结构：

```text
data/
├─ train/
├─ val/
├─ test/
├─ raw_field/
├─ masks/
└─ splits/
```

说明：

- `数据/` 保留为原始采集资料与归档目录
- `data/` 用于后续训练、验证、推理时的统一入口
- 不允许为训练方便而直接修改 `数据/` 的原有内容和目录结构

当前已建立第一阶段 smoke test 子集：

```text
data/
├─ train/
│  ├─ input/
│  └─ target/
├─ val/
│  ├─ input/
│  └─ target/
└─ DATASET_SMOKESET.md
```

说明：

- 第一阶段子集仅用于训练闭环验证
- 当前 `target` 先与 `input` 使用同图同名复制，目的是打通 dataset / trainer / checkpoint 链路
- 这不是正式监督训练数据方案
