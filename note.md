# 笔记

## 总体情况
- 数据质量
  * 信号极性：mirror 1, 2, 4, 5, 6 均为负向。
  * Mirror1 最后更新：20251009，patient_id > 315。
  * TODO：确认 auto wash 是否正确排除了全零窗口。

## 2026-04 之前的实验（精简版）

### 实验 01：ECG+rPPG → 血压估计
- 目标：从配对 ECG/rPPG 窗口端到端估计血压，按患者划分训练/验证。
- 结构：CNN + 序列建模。最新变体 01T 采用双分支 CNN（ECG/rPPG）+ Transformer + 回归头（SBP/DBP）。
- 结果：01 系列验证受限、频繁过拟合。01T 实现完成，但完整 benchmark 还没跑。

### 实验 02：ECG ↔ rPPG 跨模态互转（GAN）
- 目标：无监督双向信号转换。
- 结构：两个生成器（ECG→rPPG，rPPG→ECG）+ 两个 Patch 判别器。损失 = 对抗 + 配对 L1 + cycle L1 + identity L1。
- 结果：去掉 BP 标签过滤后跑通。快速基线（light，截断数据）：Val_pair=1.53，Val_cycle=1.13，combined=2.66。

### 实验 03：联合 Masked Reconstruction（ECG+rPPG）
- 背景：旧目标（rPPG→HR+SpO2）因标签不可靠和任务不匹配被废弃。改为自监督联合掩码重建。
- 配置：mask-aware 重建模型 + 质量加权训练，ECG 侧重中度强调（ecg_point_weight=1.25，grad_loss_weight=0.1，ecg_fft_loss_weight=0.02）。
- 最佳结果：weighted_loss=0.107，ECG_MAE=0.460，rPPG_MAE=0.240。联合 MAE=0.701。

### 实验 03-X：候选架构筛选
- 评估了 unet_gated、dual_head、tcn_ssm、cross_attention。
- 最佳：unet_gated（WeightedLoss=0.169，ECG_MAE=0.431，rPPG_MAE=0.229）。

### 实验 03-1：ECG 质量指标对比
- 比较了频域 SNR 和时域 SQI（模板相关、自相关、形态稳定性、伪差惩罚、综合评分）。
- 结论：频域 SNR 与 ECG 质量关系弱；综合时域 SQI 内部一致性更好，被选作 Exp3 权重依据。

### 实验 04：自编码器做质量/SQI
- 目标：用重建误差替代固定规则做质量建模。
- Exp4：重建误差作为质量信号。Exp4-X：三个模型直接回归 SQI。
- 最佳：exp4-2（Val MAE=0.065，Pearson=0.943，Corr(Pred,SNR)=0.928）。

---

## 2026-04 及之后的实验

## 2026-04-23
### Exp3 拆分：ECG-only / rPPG-only
- 将 Exp3 拆成两个独立单信号任务：
  * exp3_ecg：仅 ECG 掩码重建
  * exp3_rppg：仅 rPPG 掩码重建
- 参数：target_length=256，每段一个连续掩码窗口。

- 实现：
  * 共享模块：`exp3_common/single_recon_dataloader.py`、`single_recon_model.py`、`single_recon_train.py`、`single_recon_visualize.py`
  * 入口：`exp3_ecg/*`、`exp3_rppg/*`

- 训练设计：
  * 回滚后的基线 Encoder-Decoder 架构（非 U-Net 版本）
  * 输入：掩码信号 + 可见掩码拼接
  * 损失：质量加权 SmoothL1（掩码区域 + 可见区域 context loss，权重 0.2）
  * 质量权重：ECG 用自相关 SQI 排序，rPPG 用 SNR 排序

- 输出物：
  * 检查点：`checkpoints/exp3_ecg_<variant>_{best,final}.pt`、`exp3_rppg_<variant>_{best,final}.pt`
  * 图片：`exp3_ecg/plots/*`、`exp3_rppg/plots/*`

### Exp3 回滚（恢复基线）
- 背景：U-Net/额外 loss 改版训练后质量不如原基线。
- 操作：
  * 恢复 `single_recon_model.py` 为原 Encoder-Decoder 架构（light: stride-2 + residual body + transpose conv；full: stem + 3 residual block + transpose conv）
  * 恢复 `single_recon_train.py` 为纯 SmoothL1（去掉梯度/FFT 项）
- 改进：
  * 可视化脚本支持同时尝试 best 和 final 检查点，避免架构不匹配崩溃
  * 新增 `model_family="single_recon_v1"` 元数据
- 烟雾测试：full 变体可视化 MAE=0.141，训练 1 epoch val_loss=0.257。
- 建议：回滚运行加 `--checkpoint-tag _legacyv1` 避免混淆。

## 2026-04-26
### Exp3 TCN 消融
- 基线：val_loss=0.1506
- 消融 01（删 SiLU）：0.1533 ❌
- 消融 02（Dropout→Dropout1d）：0.1536 ❌
- 消融 03（target_length=512, mask=0.2）：0.1980 ❌
- 消融 04：未记录

### Exp3 Transformer
- 基线：val_loss=0.2196

### Exp3 Mamba
- 待替换为真正的 Mamba 实现
- 基线：val_loss=0.2279

### Exp3 GAN
- （空）

## 2026-05-11
### 乳酸关联分析
- 目标：探究乳酸与血压、年龄、性别、心率、血氧、呼吸率、体温的关系。
- 数据：814 条乳酸记录，245 个患者（mirror 1,2,4,5,6）。乳酸均值 1.99±0.47 mmol/L，中位数 1.94，范围 [1.02, 5.45]。

| 特征 | 与乳酸均值 | p 值 | 解读 |
|------|----------|------|------|
| 舒张压 | r=0.098 | 0.025 | 极弱正相关 |
| 收缩压 | r=0.010 | 0.827 | 无相关 |
| 年龄 | r=0.042 | 0.230 | 无相关 |
| 性别 | 男2.01, 女1.94 | 0.152 | 无显著差异 |
| 心率 | r=0.119 | 0.001 | 弱正相关 |
| 血氧 | r=0.021 | 0.574 | 无相关 |
| 呼吸率 | r=0.038 | 0.358 | 无相关 |
| 体温 | r=0.114 | 0.359 | 无相关（n=66） |

- 阈值分析：所有阈值（1.0-3.0 mmol/L）均未显示显著血压差异。最佳（仍不显著）：≥1.5 mmol/L → 舒张压 +2.43 mmHg（p=0.118）。
- 结论：**乳酸与所有现有特征几乎无相关**。最强的信号（仍很弱）是心率（r=0.119）和舒张压（r=0.098），不足以做预测。
- 可能原因：(1) 乳酸是时间敏感的急症标志物，但用的是患者平均聚合值；(2) 人群整体健康，乳酸范围窄；(3) 检验时间与体征记录可能未对齐。

## 2026-05-12
### 时序乳酸分析（Δ-Lactate）
- 目标：对齐个体乳酸测量与记录会话，检测 Δ-lactate ↔ Δ-vital 关系。
- 数据匹配：
  * 从 XLSX 解析乳酸测量：5152 条，329 个患者
  * 提取记录会话：1403 个，441 个患者
  * 时序匹配成功：850 个（60.6%），其中 ≤24h 匹配 426，≤7d 匹配 394，超 7d 匹配 30

- Δ 特征：
  * ≥2 次匹配的患者：209 个，连续会话对：598
  * 平均 Δ-lactate：0.065±0.881 mmol/L
  * Δ 时间间隔：均值 2.3 天，中位数 1.1 天

| Δ 体征 | N | r | p | 解读 |
|--------|---|----|-----|------|
| SBP | 20 | -0.543 | 0.013 | 强 |
| DBP | 20 | -0.363 | 0.116 | 不显著 |
| HR | 8 | 0.056 | 0.895 | 不显著 |
| SpO2 | 3 | — | — | 数据不足 |
| RR | 5 | 0.250 | 0.685 | 不显著 |
| 体温 | 13 | -0.377 | 0.205 | 不显著 |

- 结论：SBP 信号最强（r=-0.543），但 1/5 显著率**不可用，可能受误差值影响**。
- 与静态分析对比：静态分析几乎找不到相关（最佳 r=0.119），而 Δ 方法通过去除个体间差异发现了隐藏关系，证实了时序对齐的价值。

## 2026-06-18
### Exp3 TCN 256 课程学习
- 目标：通过逐步增加掩码难度提高 TCN 鲁棒性。
- 方法：线性课程，mask_ratio 0.10→0.30，200 epoch。
- 检查点策略：只要 val_loss ≤ best_val + 0.008 就保存。
- 训练结果：最佳 epoch=162，val_loss=0.1524，val_MAE=0.4645；最终 epoch=200，val_loss=0.1642，val_MAE=0.5002。
- 入口：`train/exp3_tcn/train_ecg_curriculum.py`。
- 检查点：`checkpoints/exp3_ecg_tcn_full_curriculum_{best,final}.pt`。

## 2026-06-19
### Exp3 TCN 512 课程学习
- 模型：TCN 512，10 层膨胀残差（kernel=5，RF≈513），544K 参数。
- 数据：5s 窗口，target_length=512（从 ~512Hz 原始数据降采样）。
- 方法：线性课程，mask_ratio 0.10→0.30，200 epoch。容忍保存（阈值 0.008）。
- 训练结果：最佳 epoch=94，val_loss=0.1182，val_MAE=0.4079（mask=0.194）；最终 epoch=200，val_loss=0.1291，val_MAE=0.4295（mask=0.30）。

- 多掩码 MSE 评估（200 个随机样本，仅掩码区域）：

| 检查点 | mask=0.10 | mask=0.20 | mask=0.35 |
|--------|:---------:|:---------:|:---------:|
| best（ep193） | 0.820 | 0.857 | 0.926 |
| final（ep200） | 0.880 | 0.850 | 0.928 |

- 观察：
  * val_loss=0.1182 是所有 Exp3 TCN 跑中最低的，远超 tcn256（0.1524）和原始 TCN（0.1506）。
  * final 在中高掩码（0.20/0.35）上略优于 best。
  * MSE 数值高于 tcn256 是因为窗口更长（点数更多），不同长度之间不可直接对比。
  * 5s 窗口带来更丰富的时序上下文，模型受益于更长的感受野。

- 文件：
  * 检查点：`checkpoints/exp3_ecg_tcn_tcn512_curriculum_{best,final}.pt`
  * 可视化：`exp3_ecg_tcn_curriculum/plots/curriculum_eval_tcn512_{best,final}.png`
  * 历史：`exp3_ecg_tcn_curriculum/plots/exp3_ecg_tcn_tcn512_curriculum_history.csv`

### 计划: 512-5s-0.10->0.40->300epoch, 对比256-3-0.1-0.4-300

## 计划: 训练ECG质量判别模型: 
首先预训练encoder. 用所有真实数据

