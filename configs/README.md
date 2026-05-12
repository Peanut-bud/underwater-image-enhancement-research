# 配置目录

配置已经从“单文件包全部参数”切换为分层结构：

```text
configs/
├─ data/
│  ├─ synth_build.yaml
│  ├─ synthetic_dataset.yaml
│  └─ real_unsup_dataset.yaml
├─ model/
│  ├─ physical_guided_base.yaml
│  └─ refinement_naf_like.yaml
├─ train/
│  ├─ supervised_base.yaml
│  └─ adapt_base.yaml
├─ infer/
│  └─ infer_base.yaml
└─ eval/
   ├─ paired_base.yaml
   └─ noref_base.yaml
```

所有配置统一使用以下顶层字段：

- `project`
- `data`
- `model`
- `training`
- `loss`
- `output`
- `runtime`

使用原则：

- `configs/data/` 只描述数据入口和阶段 1 构建规则
- `configs/model/` 只描述模型结构参数
- `configs/train/` 描述训练、损失、输出与运行时参数
- `configs/infer/` 和 `configs/eval/` 分别描述推理与评估流程
