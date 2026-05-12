# datasets 模块

本模块只负责数据读取协议，不混入训练逻辑。

当前固定入口：

- `synthetic_quad_dataset.py`
  - 读取 `input / target / transmission / airlight / metadata`
- `real_unpaired_dataset.py`
  - 读取真实无配对图像
- `infer_dataset.py`
  - 为推理入口统一列举图片
