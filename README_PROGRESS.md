# 当前进度说明

本文档用于说明截至当前版本，项目框架已经完成了哪些内容、哪些部分仍然是占位实现、以及下一步应如何继续推进。

## 1. 当前项目定位

项目主线已经从旧版的经验增强骨架切换为“四阶段物理引导增强框架”：

1. 阶段 1：物理模型合成监督数据
2. 阶段 2：物理引导增强模型
3. 阶段 3：基于四元组数据的监督预训练
4. 阶段 4：基于真实无配对图像的无监督微调

当前版本的目标不是完成最终论文算法，而是先把工程框架、配置边界、数据入口、训练入口、推理入口和测试边界全部定下来，并确保整体链路可运行。

## 2. 已完成内容

### 2.1 工程结构重构

- 顶层目录保留为 `configs / data / src / tests / input_images / outputs / checkpoints / 思路 / 数据 / 图像增强论文`
- 旧的 `highlight/frequency/diff` 主线语义已经退场
- `src/` 已按新流程拆分出 `datasets / physics / models / losses / trainers / inference / evaluators / utils`
- 训练与微调入口已经拆开，不再共用旧 trainer

### 2.2 数据入口切换

`data/` 现已切换为标准工程入口，当前包含：

- `data/raw_field/`
- `data/clean_source/`
- `data/synthetic/`
- `data/real_unsup/`
- `data/splits/`

其中：

- `data/synthetic/` 用于阶段 3 监督训练
- `data/real_unsup/` 用于阶段 4 无监督微调
- `data/raw_field/` 与 `data/clean_source/` 为阶段 1 和后续真实数据整理预留

旧的 `data/train`、`data/val` 已不再作为正式主线入口使用。目前由于工作区文件权限问题，它们在磁盘上仍存在，但已通过 `README_DEPRECATED.md` 明确标记为废弃目录。

### 2.3 模型主链搭建

当前物理引导模型主链已经建立：

- `ParameterEstimator`
  - 预测 `airlight` 和 `transmission`
- `PhysicsReconstruction`
  - 根据物理反演公式生成 `rough`
- `RefinementNet`
  - 在 `input + rough + transmission + airlight` 的拼接特征上输出最终增强结果

统一模型入口：

- [src/models/physical_guided_enhancer.py](/D:/科研/图像增强/src/models/physical_guided_enhancer.py)

统一输出字典：

```python
{
  "input": ...,
  "airlight": ...,
  "transmission": ...,
  "rough": ...,
  "enhanced": ...,
}
```

### 2.4 训练与推理入口

当前已具备以下入口：

- [src/prepare_synth.py](/D:/科研/图像增强/src/prepare_synth.py)
  - 阶段 1 的目录和配置校验入口
- [src/train_supervised.py](/D:/科研/图像增强/src/train_supervised.py)
  - 阶段 3 监督训练入口
- [src/train_adapt.py](/D:/科研/图像增强/src/train_adapt.py)
  - 阶段 4 无监督微调入口
- [src/infer.py](/D:/科研/图像增强/src/infer.py)
  - 单图与批量推理入口
- [src/eval.py](/D:/科研/图像增强/src/eval.py)
  - paired / no-reference 两类评估入口

### 2.5 损失与训练器拆分

当前损失和训练逻辑已分为两套：

- 监督训练：
  - [src/losses/supervised_losses.py](/D:/科研/图像增强/src/losses/supervised_losses.py)
  - [src/trainers/supervised_trainer.py](/D:/科研/图像增强/src/trainers/supervised_trainer.py)
- 无监督微调：
  - [src/losses/unsupervised_losses.py](/D:/科研/图像增强/src/losses/unsupervised_losses.py)
  - [src/trainers/adaptation_trainer.py](/D:/科研/图像增强/src/trainers/adaptation_trainer.py)

### 2.6 配置体系重构

`configs/` 已经从旧版扁平结构切换为分层结构：

- `configs/data/`
- `configs/model/`
- `configs/train/`
- `configs/infer/`
- `configs/eval/`

当前默认配置已经存在，可作为后续继续开发的基础。

### 2.7 测试与验证

测试已切换为新框架语义，覆盖：

- 配置加载
- 新数据集读取
- 物理层与模型前向
- 监督损失与无监督损失
- 监督训练 smoke test
- 无监督微调 smoke test
- 推理 smoke test

本轮已验证通过的命令：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:\Anaconda\envs\DL2\python.exe' -m unittest discover -s tests -p "test_*.py"
```

结果为：

```text
Ran 10 tests in 11.850s
OK
```

## 3. 当前仍是占位的部分

### 3.1 阶段 1 还没有正式完成

当前的 [src/prepare_synth.py](/D:/科研/图像增强/src/prepare_synth.py) 只负责：

- 校验目录
- 校验配置
- 汇总现有样例数量

它还没有正式完成以下功能：

- 接入真实深度估计模型
- 批量生成深度图
- 根据物理退化公式生成大规模四元组
- 自动写出完整的合成数据与元数据

### 3.2 模型仍是“最小可运行版”

当前模型能跑通，但还不是最终论文模型，仍缺少：

- 非均匀光照专门分支
- 混合曝光校正分支
- 更正式的高光/过曝恢复方案
- 更强的注意力融合设计
- 基于目标论文进一步细化的精炼子网

### 3.3 训练策略仍是“框架版”

当前监督训练和无监督微调已经可运行，但依然是骨架版：

- 数据规模很小，只用于 smoke test
- 感知损失目前是轻量替代版本
- 评估指标还不是完整论文级评估集合
- 训练超参数还未围绕真实任务做系统调优

## 4. 下一步建议

建议接下来按这个顺序继续推进：

1. 正式实现阶段 1 合成脚本
2. 将 `gemini流程.md` 中的物理退化参数范围写成配置
3. 扩充 `data/synthetic/` 为真实可训练规模
4. 在 `RefinementNet` 中逐步接入目标论文方法
5. 补齐正式评估指标和实验记录输出

## 5. 当前推荐运行环境

当前项目默认环境仍是：

- `D:\Anaconda\envs\DL2`

推荐直接调用：

```powershell
& 'D:\Anaconda\envs\DL2\python.exe' ...
```

这是因为本机上 `conda run` 当前存在编码异常，而直接使用该环境下的 Python 更稳定。
