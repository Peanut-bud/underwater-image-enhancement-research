# 生成式图像增强方案

## 1. 技术路线选择

### 1.1 方案对比

| 方案 | 数据需求 | 计算资源 | 生成质量 | 物理一致性 | 实现难度 |
|------|----------|----------|----------|------------|----------|
| **A. 轻量级生成模块** | 500-1000张 | 单卡RTX 3060 | 中等 | 高 | 低 |
| **B. GAN-based** | 1000-2000张 | 单卡RTX 3080 | 较高 | 中等 | 中等 |
| **C. 扩散模型微调** | 3000-5000张 | RTX 3090/A100 | 高 | 可控 | 高 |
| **D. ControlNet+SD** | 5000+张 | 多卡A100 | 很高 | 高 | 很高 |

### 1.2 推荐方案：A+B混合（渐进式）

**第一阶段**：轻量级生成模块（当前数据量即可）
**第二阶段**：GAN增强（数据扩充后）
**第三阶段**：扩散模型（数据充足后）

---

## 2. 第一阶段：轻量级生成模块（2-3周）

### 2.1 目标
在现有物理引导框架中加入生成式分支，提升模型对复杂退化的处理能力。

### 2.2 模型架构设计

#### 2.2.1 整体架构
```
输入退化图 (I)
    ↓
┌─────────────────────────────────────────┐
│  ParameterEstimator (保留原有)          │
│  - 预测 airlight (A)                    │
│  - 预测 transmission (t)                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  PhysicsReconstruction (保留原有)       │
│  - 物理反演: rough = (I - A) / t + A    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  双分支 RefinementNet (核心改进)        │
│  ┌─────────────────────────────────┐    │
│  │ 分支1: 确定性分支 (NAF块)       │    │
│  │ - 保留原有精炼能力              │    │
│  │ - 保证基础质量                  │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │ 分支2: 生成式分支 (新增)        │    │
│  │ - 残差生成网络                  │    │
│  │ - 注意力机制                    │    │
│  │ - 增强细节恢复                  │    │
│  └─────────────────────────────────┘    │
│  ┌─────────────────────────────────┐    │
│  │ 自适应融合模块                  │    │
│  │ - 物理一致性约束                │    │
│  │ - 动态权重学习                  │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
    ↓
输出增强图 (J)
```

#### 2.2.2 详细模块设计

**模块1：生成式分支 (GenerativeBranch)**
```python
class GenerativeBranch(nn.Module):
    """轻量级生成式分支，用于增强细节恢复"""

    def __init__(self, in_channels: int = 10, hidden_channels: int = 32):
        super().__init__()
        # 多尺度特征提取
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=5, padding=2),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=7, padding=3),
        ])

        # 注意力机制
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels * 3, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels * 3, 1),
            nn.Sigmoid()
        )

        # 残差生成
        self.residual_generator = nn.Sequential(
            nn.Conv2d(hidden_channels * 3, hidden_channels * 2, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_channels, 3, 3, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]，之后缩放到 [0, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # 多尺度特征提取
        multi_features = [conv(features) for conv in self.multi_scale]
        concat_features = torch.cat(multi_features, dim=1)

        # 注意力加权
        attention_weights = self.attention(concat_features)
        weighted_features = concat_features * attention_weights

        # 生成残差
        residual = self.residual_generator(weighted_features)
        return residual * 0.1  # 缩小残差，避免过度生成
```

**模块2：物理一致性融合 (PhysicsConsistentFusion)**
```python
class PhysicsConsistentFusion(nn.Module):
    """融合确定性分支和生成式分支，保持物理一致性"""

    def __init__(self, channels: int = 32):
        super().__init__()
        # 动态权重学习
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.ReLU(),
            nn.Conv2d(channels, 2, 1),
            nn.Softmax(dim=1)
        )

        # 物理一致性校验
        self.physics_checker = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1),  # 输入: rough + generated
            nn.ReLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()  # 物理一致性得分
        )

    def forward(
        self,
        deterministic: torch.Tensor,
        generative: torch.Tensor,
        rough: torch.Tensor,
        airlight: torch.Tensor,
        transmission: torch.Tensor
    ) -> torch.Tensor:
        # 计算动态权重
        combined = torch.cat([deterministic, generative], dim=1)
        weights = self.weight_net(combined)
        w_det, w_gen = weights[:, 0:1], weights[:, 1:2]

        # 初始融合
        fused = w_det * deterministic + w_gen * generative

        # 物理一致性约束
        # 生成的图应该满足: generated ≈ rough * t + A * (1 - t)
        expected = rough * transmission + airlight * (1 - transmission)
        physics_score = self.physics_checker(torch.cat([fused, expected], dim=1))

        # 根据物理一致性得分调整融合
        final = fused * physics_score + rough * (1 - physics_score)

        return final.clamp(0, 1)
```

**模块3：增强版RefinementNet**
```python
class EnhancedRefinementNet(nn.Module):
    """增强版精炼网络，融合确定性和生成式分支"""

    def __init__(
        self,
        in_channels: int = 10,
        hidden_channels: int = 32,
        blocks: int = 2,
        residual_scale: float = 0.2
    ):
        super().__init__()
        self.residual_scale = residual_scale

        # 原有确定性分支
        self.deterministic_branch = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            *[NAFLikeBlock(hidden_channels) for _ in range(blocks)],
            nn.Conv2d(hidden_channels, 3, 3, padding=1),
            nn.Tanh()
        )

        # 新增生成式分支
        self.generative_branch = GenerativeBranch(in_channels, hidden_channels)

        # 物理一致性融合
        self.fusion = PhysicsConsistentFusion(hidden_channels)

        # 最终输出
        self.final_head = nn.Sequential(
            nn.Conv2d(3, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        rough: torch.Tensor,
        features: torch.Tensor,
        airlight: torch.Tensor,
        transmission: torch.Tensor
    ) -> torch.Tensor:
        # 确定性分支
        det_residual = self.deterministic_branch(features)
        det_output = rough + self.residual_scale * det_residual

        # 生成式分支
        gen_residual = self.generative_branch(features)
        gen_output = rough + self.residual_scale * gen_residual

        # 物理一致性融合
        enhanced = self.fusion(det_output, gen_output, rough, airlight, transmission)

        return self.final_head(enhanced)
```

### 2.3 损失函数设计

#### 2.3.1 新增损失组件

**1. 生成式分支正则化损失**
```python
class GenerativeRegularizationLoss(nn.Module):
    """防止生成式分支过度偏离物理约束"""

    def __init__(self, lambda_physics: float = 0.5, lambda_sparsity: float = 0.1):
        super().__init__()
        self.lambda_physics = lambda_physics
        self.lambda_sparsity = lambda_sparsity

    def forward(
        self,
        generated: torch.Tensor,
        rough: torch.Tensor,
        airlight: torch.Tensor,
        transmission: torch.Tensor
    ) -> torch.Tensor:
        # 物理一致性损失
        expected = rough * transmission + airlight * (1 - transmission)
        physics_loss = F.l1_loss(generated, expected)

        # 稀疏性损失（鼓励生成式分支只在需要的地方激活）
        sparsity_loss = generated.abs().mean()

        return self.lambda_physics * physics_loss + self.lambda_sparsity * sparsity_loss
```

**2. 多尺度感知损失**
```python
class MultiScalePerceptualLoss(nn.Module):
    """多尺度感知损失，增强生成质量"""

    def __init__(self, scales: list[int] = [1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.perceptual = VGGPerceptualLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for scale in self.scales:
            if scale == 1:
                pred_scaled = pred
                target_scaled = target
            else:
                pred_scaled = F.avg_pool2d(pred, kernel_size=scale)
                target_scaled = F.avg_pool2d(target, kernel_size=scale)

            total_loss += self.perceptual(pred_scaled, target_scaled)

        return total_loss / len(self.scales)
```

#### 2.3.2 增强版损失函数

```python
class EnhancedSupervisedLossWeights:
    # 原有损失
    recon: float = 1.0
    ssim: float = 0.5
    edge: float = 0.2
    transmission: float = 0.5
    airlight: float = 0.2
    perceptual: float = 0.1

    # 新增损失
    generative_reg: float = 0.3
    multi_scale_perceptual: float = 0.2
    physics_consistency: float = 0.4

class EnhancedSupervisedLossStack(nn.Module):
    def __init__(self, weights: EnhancedSupervisedLossWeights):
        super().__init__()
        self.weights = weights
        # 原有损失
        self.ssim = SSIMLoss()
        self.edge = EdgeLoss()
        self.perceptual = VGGPerceptualLoss()
        # 新增损失
        self.generative_reg = GenerativeRegularizationLoss()
        self.multi_scale_perceptual = MultiScalePerceptualLoss()
        self.physics_consistency = PhysicsConsistencyLoss()

    def forward(self, outputs: dict, batch: dict) -> dict:
        enhanced = outputs["enhanced"]
        target = batch["target"]
        rough = outputs["rough"]
        airlight = outputs["airlight"]
        transmission = outputs["transmission"]

        # 原有损失
        recon_loss = F.l1_loss(enhanced, target)
        ssim_loss = self.ssim(enhanced, target)
        edge_loss = self.edge(enhanced, target)
        transmission_loss = F.mse_loss(transmission, batch["transmission"])
        airlight_loss = F.mse_loss(airlight, batch["airlight"])
        perceptual_loss = self.perceptual(enhanced, target)

        # 新增损失
        generative_reg_loss = self.generative_reg(
            outputs["generative_output"], rough, airlight, transmission
        )
        multi_scale_loss = self.multi_scale_perceptual(enhanced, target)
        physics_loss = self.physics_consistency(enhanced, rough, airlight, transmission)

        # 总损失
        total = (
            self.weights.recon * recon_loss
            + self.weights.ssim * ssim_loss
            + self.weights.edge * edge_loss
            + self.weights.transmission * transmission_loss
            + self.weights.airlight * airlight_loss
            + self.weights.perceptual * perceptual_loss
            + self.weights.generative_reg * generative_reg_loss
            + self.weights.multi_scale_perceptual * multi_scale_loss
            + self.weights.physics_consistency * physics_loss
        )

        return {
            "total": total,
            "recon": recon_loss,
            "ssim": ssim_loss,
            "edge": edge_loss,
            "transmission": transmission_loss,
            "airlight": airlight_loss,
            "perceptual": perceptual_loss,
            "generative_reg": generative_reg_loss,
            "multi_scale_perceptual": multi_scale_loss,
            "physics_consistency": physics_loss,
        }
```

### 2.4 训练策略

#### 2.4.1 渐进式训练
```python
def progressive_training(model, train_loader, val_loader, config):
    """渐进式训练策略"""

    # 阶段1：冻结生成式分支，训练确定性分支（5 epochs）
    print("Stage 1: Training deterministic branch only")
    freeze_module(model.refinement_net.generative_branch)
    train_stage(model, train_loader, val_loader, epochs=5, lr=1e-3)

    # 阶段2：解冻生成式分支，联合训练（10 epochs）
    print("Stage 2: Joint training with generative branch")
    unfreeze_module(model.refinement_net.generative_branch)
    train_stage(model, train_loader, val_loader, epochs=10, lr=5e-4)

    # 阶段3：微调整个网络（5 epochs）
    print("Stage 3: Fine-tuning entire network")
    train_stage(model, train_loader, val_loader, epochs=5, lr=1e-4)
```

#### 2.4.2 学习率调度
```python
def get_optimizer(model, config):
    """分层学习率优化器"""

    # 基础层（物理参数估计、物理重建）使用较小学习率
    base_params = list(model.parameter_estimator.parameters()) + \
                  list(model.physics_reconstruction.parameters())

    # 精炼层使用较大学习率
    refinement_params = list(model.refinement_net.parameters())

    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': config.lr * 0.1},
        {'params': refinement_params, 'lr': config.lr}
    ], weight_decay=1e-4)

    return optimizer
```

### 2.5 数据增强策略

```python
class EnhancedAugmentation:
    """针对生成式增强的数据增强策略"""

    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        input_img = sample['input']
        target_img = sample['target']

        # 1. 随机裁剪（增强局部细节学习）
        if random.random() > 0.5:
            crop_size = random.choice([128, 192, 224])
            input_img, target_img = random_crop_pair(input_img, target_img, crop_size)

        # 2. 随机翻转
        if random.random() > 0.5:
            input_img = torch.flip(input_img, dims=[2])
            target_img = torch.flip(target_img, dims=[2])

        # 3. 颜色抖动（增强颜色鲁棒性）
        if random.random() > 0.7:
            input_img = color_jitter(input_img, brightness=0.2, contrast=0.2)

        # 4. 随机噪声（增强噪声鲁棒性）
        if random.random() > 0.8:
            noise_level = random.uniform(0.01, 0.05)
            input_img = add_random_noise(input_img, noise_level)

        return {'input': input_img, 'target': target_img}
```

---

## 3. 第二阶段：GAN增强（1-2个月后）

### 3.1 条件GAN架构

```python
class ConditionalGenerator(nn.Module):
    """条件生成器，基于物理约束生成增强图像"""

    def __init__(self, in_channels=10, out_channels=3):
        super().__init__()
        # U-Net结构
        self.encoder = nn.ModuleList([
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
        ])

        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1),
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
        ])

        # 跳跃连接
        self.skip_connections = True

    def forward(self, x, rough, airlight, transmission):
        # 编码
        enc_features = []
        for enc in self.encoder:
            x = F.leaky_relu(enc(x), 0.2)
            enc_features.append(x)

        # 解码
        for i, dec in enumerate(self.decoder):
            x = F.relu(dec(x))
            if self.skip_connections and i < len(enc_features):
                x = torch.cat([x, enc_features[-(i+1)]], dim=1)

        # 物理约束
        x = torch.sigmoid(x)
        # 确保输出满足物理模型
        x = x * transmission + airlight * (1 - transmission)

        return x

class PatchDiscriminator(nn.Module):
    """PatchGAN判别器"""

    def __init__(self, in_channels=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))
```

### 3.2 GAN训练策略

```python
class GANTrainer:
    """GAN训练器"""

    def __init__(self, generator, discriminator, config):
        self.generator = generator
        self.discriminator = discriminator
        self.config = config

        # 损失函数
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()

    def train_step(self, batch):
        input_img = batch['input']
        target_img = batch['target']
        rough = batch['rough']
        airlight = batch['airlight']
        transmission = batch['transmission']

        # 训练判别器
        fake_img = self.generator(input_img, rough, airlight, transmission)

        real_pred = self.discriminator(input_img, target_img)
        fake_pred = self.discriminator(input_img, fake_img.detach())

        d_loss_real = self.adversarial_loss(real_pred, torch.ones_like(real_pred))
        d_loss_fake = self.adversarial_loss(fake_pred, torch.zeros_like(fake_pred))
        d_loss = (d_loss_real + d_loss_fake) / 2

        # 训练生成器
        fake_pred = self.discriminator(input_img, fake_img)
        g_loss_adv = self.adversarial_loss(fake_pred, torch.ones_like(fake_pred))
        g_loss_rec = self.reconstruction_loss(fake_img, target_img)
        g_loss_per = self.perceptual_loss(fake_img, target_img)

        g_loss = g_loss_adv + 100 * g_loss_rec + 10 * g_loss_per

        return d_loss, g_loss
```

---

## 4. 第三阶段：扩散模型（3-6个月后）

### 4.1 ControlNet集成

```python
class PhysicsControlNet(nn.Module):
    """物理约束的ControlNet"""

    def __init__(self):
        super().__init__()
        # 深度条件（透射率图）
        self.depth_condition = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        # 颜色条件（大气光图）
        self.color_condition = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 1)
        )

    def forward(self, x, transmission, airlight):
        depth_feat = self.depth_condition(transmission)
        color_feat = self.color_condition(airlight)
        combined = torch.cat([x, depth_feat, color_feat], dim=1)
        return self.fusion(combined)
```

### 4.2 扩散模型微调策略

```python
def fine_tune_diffusion_model(pretrained_model, physics_controlnet, dataset):
    """微调预训练的扩散模型"""

    # 冻结大部分预训练参数
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # 只训练ControlNet和最后几层
    for param in physics_controlnet.parameters():
        param.requires_grad = True

    # 微调配置
    optimizer = torch.optim.AdamW(
        list(physics_controlnet.parameters()) +
        list(pretrained_model.out.parameters()),
        lr=1e-5,
        weight_decay=1e-2
    )

    # 训练循环
    for epoch in range(100):
        for batch in dataset:
            # 使用物理条件引导生成
            noise = torch.randn_like(batch['target'])
            timesteps = torch.randint(0, 1000, (batch['target'].shape[0],))

            # 前向扩散
            noisy_images = add_noise(batch['target'], noise, timesteps)

            # 条件生成
            condition = physics_controlnet(
                batch['input'],
                batch['transmission'],
                batch['airlight']
            )

            # 预测噪声
            predicted_noise = pretrained_model(noisy_images, timesteps, condition)

            # 损失
            loss = F.mse_loss(predicted_noise, noise)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 5. 实施路线图

### 5.1 第一阶段（2-3周）

**Week 1：模型架构实现**
- [ ] 实现 `GenerativeBranch` 模块
- [ ] 实现 `PhysicsConsistentFusion` 模块
- [ ] 实现 `EnhancedRefinementNet` 模块
- [ ] 集成到现有 `PhysicalGuidedEnhancer`

**Week 2：损失函数与训练**
- [ ] 实现新增损失函数
- [ ] 修改 `EnhancedSupervisedLossStack`
- [ ] 实现渐进式训练策略
- [ ] 更新配置文件

**Week 3：测试与优化**
- [ ] 单元测试
- [ ] Smoke test
- [ ] 性能基准测试
- [ ] 超参数调优

### 5.2 第二阶段（1-2个月后）

**Month 1：GAN架构实现**
- [ ] 实现 `ConditionalGenerator`
- [ ] 实现 `PatchDiscriminator`
- [ ] 实现GAN训练器
- [ ] 数据扩充策略

**Month 2：GAN训练与调优**
- [ ] GAN训练流程
- [ ] 模式崩溃检测
- [ ] 稳定性优化
- [ ] 评估指标

### 5.3 第三阶段（3-6个月后）

**Month 3-4：扩散模型集成**
- [ ] ControlNet实现
- [ ] 预训练模型加载
- [ ] 物理条件编码

**Month 5-6：扩散模型微调**
- [ ] 微调策略实现
- [ ] 推理优化
- [ ] 质量评估

---

## 6. 数据需求与准备

### 6.1 当前数据状态
- 总图片数：2500+张
- 分辨率：主要1920×1080，部分其他分辨率
- 场景：多个现场测试，差异较大

### 6.2 数据准备建议

**立即需要（第一阶段）**
1. 整理 `数据/` 目录到 `data/raw_field/`
2. 统一分辨率（建议256×256或512×512）
3. 生成更多合成数据（建议5000+张）

**后续需要（第二、三阶段）**
1. 数据扩充到10000+张
2. 增加场景多样性
3. 收集更多退化类型

### 6.3 数据扩充策略

```python
class DataAugmentationPipeline:
    """数据扩充流水线"""

    def __init__(self, config):
        self.config = config

    def augment_dataset(self, input_dir, output_dir, target_count=5000):
        """扩充数据集到目标数量"""

        # 1. 基础增强（翻转、旋转、裁剪）
        basic_augmented = self.basic_augmentation(input_dir)

        # 2. 物理参数扰动
        physics_augmented = self.physics_perturbation(basic_augmented)

        # 3. 退化类型混合
        mixed_augmented = self.degradation_mixing(physics_augmented)

        # 4. 颜色空间变换
        color_augmented = self.color_transform(mixed_augmented)

        return color_augmented

    def basic_augmentation(self, images):
        """基础几何增强"""
        augmented = []
        for img in images:
            # 原图
            augmented.append(img)
            # 水平翻转
            augmented.append(torch.flip(img, dims=[2]))
            # 垂直翻转
            augmented.append(torch.flip(img, dims=[1]))
            # 90度旋转
            augmented.append(torch.rot90(img, k=1, dims=[1, 2]))
            # 180度旋转
            augmented.append(torch.rot90(img, k=2, dims=[1, 2]))
        return augmented

    def physics_perturbation(self, samples):
        """物理参数扰动"""
        augmented = []
        for sample in samples:
            # 原始参数
            A = sample['airlight']
            t = sample['transmission']
            beta = sample['beta']

            # 扰动范围
            for _ in range(3):
                A_new = A + torch.randn_like(A) * 0.05
                beta_new = beta * (1 + torch.randn(1) * 0.2)
                t_new = torch.exp(-beta_new * sample['depth'])

                # 生成新的退化图像
                I_new = sample['target'] * t_new + A_new * (1 - t_new)

                augmented.append({
                    'input': I_new,
                    'target': sample['target'],
                    'airlight': A_new,
                    'transmission': t_new,
                    'beta': beta_new
                })

        return augmented
```

---

## 7. 风险控制

### 7.1 过度生成风险

**问题**：生成式分支可能产生不存在的细节（幻觉）

**解决方案**：
1. 物理一致性损失约束
2. 残差缩放因子（0.1-0.2）
3. 生成式分支正则化
4. 渐进式训练（先确定性，后生成式）

```python
def prevent_hallucination(outputs, batch):
    """防止过度生成的检查机制"""

    # 1. 物理一致性检查
    physics_error = compute_physics_error(
        outputs['enhanced'],
        outputs['rough'],
        outputs['airlight'],
        outputs['transmission']
    )

    # 2. 边缘保持检查
    edge_similarity = compute_edge_similarity(
        outputs['enhanced'],
        batch['input']
    )

    # 3. 颜色分布检查
    color_consistency = compute_color_consistency(
        outputs['enhanced'],
        batch['target']
    )

    # 综合得分
    consistency_score = (
        0.4 * physics_error +
        0.3 * edge_similarity +
        0.3 * color_consistency
    )

    return consistency_score
```

### 7.2 训练稳定性风险

**问题**：GAN训练可能不稳定，出现模式崩溃

**解决方案**：
1. 使用WGAN-GP损失
2. 渐进式训练
3. 学习率调度
4. 梯度裁剪

```python
class StableGANLoss:
    """稳定的GAN损失函数"""

    def __init__(self, lambda_gp=10):
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, discriminator, real, fake, condition):
        """梯度惩罚"""
        alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

        d_interpolated = discriminator(condition, interpolated)

        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
```

### 7.3 计算资源风险

**问题**：扩散模型需要大量计算资源

**解决方案**：
1. 渐进式实现（轻量级→GAN→扩散）
2. 模型压缩技术
3. 推理优化
4. 云端训练

---

## 8. 评估指标

### 8.1 客观指标

**图像质量指标**
- PSNR（峰值信噪比）
- SSIM（结构相似性）
- LPIPS（感知相似性）
- NIQE（自然图像质量）

**物理一致性指标**
- 透射率误差
- 大气光误差
- 物理模型重建误差

**生成质量指标**
- FID（Fréchet Inception Distance）
- IS（Inception Score）
- LPIPS多样性

### 8.2 主观评估

**用户研究**
- 盲测对比（当前方法 vs 生成式方法）
- 5分制质量评分
- 细节保持评分
- 自然度评分

---

## 9. 配置文件设计

### 9.1 模型配置

```yaml
# configs/model/enhanced_generative.yaml
model:
  name: "enhanced_generative_enhancer"
  base_channels: 32
  t_min: 0.05
  refinement_blocks: 2

  # 生成式分支配置
  generative_branch:
    enabled: true
    hidden_channels: 32
    residual_scale: 0.15
    multi_scale_kernels: [3, 5, 7]
    attention_reduction: 4

  # 融合模块配置
  fusion:
    type: "physics_consistent"
    physics_weight: 0.4
    dynamic_weight: true

  # 预训练权重
  pretrained: "./checkpoints/physical_guided_supervised_latest.pth"
```

### 9.2 训练配置

```yaml
# configs/train/enhanced_generative.yaml
training:
  epochs: 20
  batch_size: 4
  learning_rate: 5e-4
  weight_decay: 1e-4

  # 渐进式训练
  progressive:
    enabled: true
    stages:
      - name: "deterministic"
        epochs: 5
        freeze_generative: true
        lr: 1e-3
      - name: "joint"
        epochs: 10
        freeze_generative: false
        lr: 5e-4
      - name: "finetune"
        epochs: 5
        freeze_generative: false
        lr: 1e-4

  # 学习率调度
  scheduler:
    type: "cosine"
    T_max: 20
    eta_min: 1e-6

  # 混合精度
  amp: true

  # 梯度裁剪
  grad_clip: 1.0
```

### 9.3 损失函数配置

```yaml
# configs/loss/enhanced_generative.yaml
loss:
  # 原有损失
  recon: 1.0
  ssim: 0.5
  edge: 0.2
  transmission: 0.5
  airlight: 0.2
  perceptual: 0.1

  # 新增损失
  generative_reg: 0.3
  multi_scale_perceptual: 0.2
  physics_consistency: 0.4

  # 感知损失配置
  perceptual_config:
    use_stub: false
    layers: ["relu1_2", "relu2_2", "relu3_3"]
    weights: [1.0, 1.0, 1.0]
```

---

## 10. 总结

### 10.1 技术优势

1. **渐进式实现**：降低风险，便于调试
2. **物理约束**：保持可解释性，避免幻觉
3. **模块化设计**：易于扩展和替换
4. **兼容现有框架**：最小化改动

### 10.2 预期效果

**第一阶段（轻量级生成模块）**
- 细节恢复提升10-20%
- 训练时间增加约30%
- 推理时间增加约15%

**第二阶段（GAN增强）**
- 生成质量提升30-50%
- 需要更多训练数据
- 计算成本增加2-3倍

**第三阶段（扩散模型）**
- 生成质量接近GPT级别
- 需要大量计算资源
- 推理时间较长

### 10.3 下一步行动

1. **立即开始**：实现第一阶段轻量级生成模块
2. **数据准备**：整理现有数据，扩充合成数据
3. **资源评估**：确定可用计算资源
4. **时间规划**：根据资源调整实施计划

---

**文档版本**：v1.0
**最后更新**：2026-05-12
**作者**：AI Assistant
