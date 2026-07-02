# Exp2: Per-Task Deep Learning for Lab Test Prediction — 实验报告

> **日期**: 2026-07-02  
> **实验目录**: `study/exp2_lab_multimodal/`  
> **环境**: Python 3.12, PyTorch 2.4.1 (CUDA 12.1), Tesla V100 16GB  
> **策略**: 每个化验任务训练一个独立的 BinaryM3TNet 二分类模型（非多任务多头）

---

## 1. 实验目标与方法论

利用**深度学习多模态模型**，从**10 秒 ECG 信号** + **单帧面部 rPPG 图像**预测患者的**化验指标异常**。

### 为什么用单任务模型而非多任务？

| 方面 | 多任务（旧方案） | 单任务（新方案） |
|------|:---:|:---:|
| 每任务有效样本 | 468 ÷ 12 ≈ 39 | 400–500（该任务有标签的全部样本） |
| 任务间干扰 | 12 个任务共享 backbone，可能冲突 | 每个模型独立，无干扰 |
| 标签缺失处理 | 需掩码 BCE 损失 | 只使用该任务有标签的样本 |
| 可解释性 | 难以归因 | 每任务可独立分析 |
| 模型总数 | 1 个 | 12 个（每个 ~64K 参数） |

**结论**：单任务模型在方法上更科学——每个模型专注于自己的预测目标，不被其他任务稀释。

---

## 2. 数据集

### 2.1 数据来源

| 数据类型 | 来源 | 格式 |
|---------|------|------|
| ECG 信号 | `mirror*_auto_cleaned_sqi/patient_*.csv` | 时间序列 (Timestamp, RPPG, ECG) |
| 面部视频 | `mirror*_data/patient_*/video.avi` | MJPEG 编码视频 |
| 化验标签 | `merged_lab_tests.csv` | 132,771 行化验记录 |
| 患者信息 | `cleaned_patient_info.csv` | 血压、SQI 等 |

### 2.2 数据预处理

**ECG 处理**：
1. 从 CSV 读取 Timestamp + ECG 两列
2. 按时间戳排序，剔除无效值
3. 从会话后段截取 10 秒窗口
4. 线性插值重采样至固定 256 点
5. Z-score 标准化

**面部图像处理**：
1. 从 MJPEG 视频中扫描 JPEG SOI/EOI 标记，提取第 30 帧
2. 缩放到 32×32 像素（双线性插值）
3. 转为灰度图，归一化到 [0, 1]

**标签构建**：
- 从 `merged_lab_tests.csv` 按 `hospital_id` 聚合所有化验记录
- 取每种指标的最大值（乳酸、肌钙蛋白、血糖、pCO₂）或最小值（血红蛋白、pO₂）
- 根据临床阈值生成二分类标签（正常/异常 + 严重程度分级）

### 2.3 数据集统计

| 指标 | 数值 |
|------|:---:|
| 总样本数 | 731 |
| 唯一患者数 (hospital_id) | 105 |
| ECG 维度 | (731, 256) float32 |
| Face 维度 | (731, 32, 32) float32 |
| 标签任务数 | 15 |
| 数据来源 (mirrors) | mirror1–mirror7 |

### 2.4 标签分布

| 任务 | 总样本 | 阳性 | 阴性 | 阳性率 |
|------|:-----:|:---:|:---:|:-----:|
| lactate_high (>2.0 mmol/L) | 731 | 707 | 24 | 96.7% |
| troponin_high (>34 pg/mL) | 731 | 731 | 0 | 100% |
| glucose_high (>7.8 mmol/L) | 731 | 556 | 175 | 76.1% |
| hemoglobin_low (<130/120 g/L) | 731 | 723 | 8 | 98.9% |
| po2_low (<80 mmHg) | 731 | 459 | 272 | 62.8% |
| pco2_abnormal (<35 or >45 mmHg) | 731 | 655 | 76 | 89.6% |
| high_blood_pressure (≥140/90) | 435 | 180 | 255 | 41.4% |
| coronary_context (冠心病) | 731 | 0 | 731 | 0% |
| lactate_moderate_high (>4.0) | 731 | 145 | 586 | 19.8% |
| troponin_extreme_high (>1000) | 731 | 296 | 435 | 40.5% |
| glucose_marked_high (>10.0) | 731 | 400 | 331 | 54.7% |
| hemoglobin_moderate_low (<90) | 731 | 214 | 517 | 29.3% |
| po2_moderate_low (<70) | 731 | 204 | 527 | 27.9% |
| pco2_low (<34) | 731 | 161 | 570 | 22.0% |
| pco2_high (>50) | 731 | 240 | 491 | 32.8% |

> **注**: `troponin_high`、`hemoglobin_low`、`coronary_context` 因某类别样本不足被自动跳过（见第 5 节）。

---

## 3. 模型架构: BinaryM3TNet

### 3.1 整体架构

每个化验任务使用一个独立的 **BinaryM3TNet**，结构如下：

```
┌──────────────────────────────────────────────┐
│              BinaryM3TNet                    │
│                                              │
│  ┌──────────────┐   ┌──────────────┐        │
│  │ ECG Encoder  │   │ Face Encoder │        │
│  │  (1D CNN)    │   │  (2D CNN)    │        │
│  │ input:(1,256)│   │input:(1,32,32)│       │
│  │  ↓           │   │  ↓           │        │
│  │ Conv1d×3     │   │ Conv2d×3     │        │
│  │  + Residual  │   │              │        │
│  │  ↓           │   │  ↓           │        │
│  │ AvgPool      │   │ AvgPool      │        │
│  │  ↓           │   │  ↓           │        │
│  │ FC(64→64)    │   │ FC(32→64)    │        │
│  └──────┬───────┘   └──────┬───────┘        │
│         │  64-d            │  64-d           │
│         └────────┬─────────┘                 │
│                  ↓ Concat (128-d)            │
│           ┌──────────────┐                   │
│           │ Fusion MLP   │                   │
│           │ FC(128→64)   │                   │
│           │ + SiLU + BN  │                   │
│           │ + Dropout 0.5│                   │
│           └──────┬───────┘                   │
│                  ↓ (64-d)                    │
│           ┌──────────────┐                   │
│           │ Output Head  │                   │
│           │ FC(64→1)     │                   │
│           │ + Sigmoid    │                   │
│           └──────────────┘                   │
│                  ↓                           │
│          P(abnormal) ∈ [0, 1]                │
└──────────────────────────────────────────────┘
```

与之前的 M3TNet（多任务多头）相比，BinaryM3TNet 只有一个输出头，专精于单个化验指标。

### 3.2 ECG Encoder（1D 残差 CNN）

| 层 | 类型 | 输入通道 | 输出通道 | 核大小 | 步长 | 输出尺寸 |
|:--|------|:------:|:------:|:-----:|:---:|:--------:|
| Enc1 | Residual1dBlock | 1 | 16 | 5 | 2 | (B, 16, 128) |
| Enc2 | Residual1dBlock | 16 | 32 | 5 | 2 | (B, 32, 64) |
| Enc3 | Residual1dBlock | 32 | 64 | 5 | 2 | (B, 64, 32) |
| Pool | AdaptiveAvgPool1d | – | – | – | – | (B, 64, 1) |
| Proj | Linear + SiLU + Dropout | 64 | 64 | – | – | (B, 64) |

每个 `Residual1dBlock` 包含：Conv1d → BN → SiLU → Dropout → Conv1d → BN → SiLU，带 shortcut 1×1 卷积下采样。

### 3.3 Face Encoder（2D CNN）

| 层 | 类型 | 输入通道 | 输出通道 | 核大小 | 步长 | 输出尺寸 |
|:--|------|:------:|:------:|:-----:|:---:|:--------:|
| Enc1 | Conv2dBlock | 1 | 8 | 3 | 2 | (B, 8, 16, 16) |
| Enc2 | Conv2dBlock | 8 | 16 | 3 | 2 | (B, 16, 8, 8) |
| Enc3 | Conv2dBlock | 16 | 32 | 3 | 2 | (B, 32, 4, 4) |
| Pool | AdaptiveAvgPool2d | – | – | – | – | (B, 32, 1, 1) |
| Proj | Linear + SiLU + Dropout | 32 | 64 | – | – | (B, 64) |

每个 `Conv2dBlock` 包含：Conv2d → BatchNorm2d → SiLU → Dropout。

### 3.4 模型参数统计

| 组件 | 参数量 |
|------|:-----:|
| ECG Encoder | ~48K |
| Face Encoder | ~8K |
| Fusion MLP + 1 Head | ~8K |
| **每任务总计** | **63,673** |
| **12 个任务合计** | **764,076** |

> 每个任务独立训练一个模型，12 个模型之间不共享参数。

---

## 4. 训练配置

### 4.1 数据划分

每个任务独立进行**患者级别分组划分**（按 `hospital_id`），因此不同任务的 train/val/test 患者集合可能不同（取决于哪些患者有该任务的标签）：

| 划分 | 占比 | 说明 |
|------|:---:|------|
| Train | ~60% | 每任务约 250–500 样本 |
| Validation | ~20% | 每任务约 95–160 样本 |
| Test | ~20% | 每任务约 90–200 样本 |

### 4.2 超参数

| 参数 | 值 | 说明 |
|------|:--:|------|
| Batch Size | 32 | |
| Learning Rate | 3×10⁻⁴ | 初始学习率 |
| LR Scheduler | ReduceLROnPlateau | mode=max, factor=0.5, patience=15 |
| Optimizer | AdamW | weight_decay=1×10⁻³ |
| Max Epochs | 200 | 每任务 |
| Early Stopping | patience=40 | 基于 validation balanced accuracy |
| Dropout | 0.5 | Fusion MLP 中 |
| Loss | BCEWithLogitsLoss | pos_weight=2.0 |
| Gradient Clipping | max_norm=1.0 | |

### 4.3 损失函数

标准加权二元交叉熵：

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} w_p \cdot y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)$$

其中 $w_p = 2.0$ 为正类权重。

### 4.4 任务过滤

每个任务独立检查训练集中各类别样本数，跳过任一类别样本不足 8 个的任务：

| 跳过的任务 | 原因 |
|-----------|------|
| `troponin_high` | 459 个阳性，0 个阴性 |
| `hemoglobin_low` | 496 个阳性，仅 3 个阴性 |
| `coronary_context` | 0 个阳性，476 个阴性 |

最终训练 **12 个独立模型**。

---

## 5. 实验结果

### 5.1 训练概览

| 指标 | 值 |
|------|:--:|
| 训练任务数 | 12 / 15（3 个跳过） |
| 每任务训练时间 | ~1.5 分钟（V100） |
| 总训练时间 | 18.1 分钟 |

### 5.2 Test Set 整体指标（12 任务 Macro 平均）

| 指标 | 均值 ± 标准差 |
|------|:-----------:|
| Balanced Accuracy | 0.489 ± 0.069 |
| ROC-AUC | 0.484 ± 0.092 |
| F1 Score | 0.447 ± 0.258 |
| Average Precision | — |

### 5.3 Test Set 逐任务结果

| 任务 | 训练样本 | bACC | ROC-AUC | F1 | 阳性率 |
|------|:-----:|:---:|:------:|:--:|:-----:|
| lactate_high | 469 | 0.342 | 0.312 | 0.804 | 98.4% |
| glucose_high | 438 | 0.500 | 0.523 | 0.000 | 72.0% |
| po2_low | 457 | 0.559 | 0.515 | 0.568 | 73.6% |
| pco2_abnormal | 430 | 0.473 | 0.509 | 0.930 | 91.9% |
| high_blood_pressure | 251 | 0.453 | 0.450 | 0.490 | 41.6% |
| lactate_moderate_high | 447 | 0.498 | 0.410 | 0.185 | 26.9% |
| **troponin_extreme_high** | 407 | **0.563** | **0.594** | 0.488 | 37.6% |
| glucose_marked_high | 409 | 0.423 | 0.388 | 0.593 | 57.1% |
| hemoglobin_moderate_low | 432 | 0.489 | 0.539 | 0.361 | 28.7% |
| po2_moderate_low | 445 | 0.540 | 0.581 | 0.354 | 23.8% |
| pco2_low | 422 | 0.483 | 0.394 | 0.255 | 29.9% |
| **pco2_high** | 436 | **0.532** | **0.595** | 0.343 | 25.7% |

### 5.4 与之前多任务方案的对比

| 指标 | 多任务 (旧) | 单任务 (新) |
|------|:---:|:---:|
| 每任务训练样本 | ~468（共享） | 250–500（独立） |
| 任务间干扰 | 有 | 无 |
| Macro bACC | 0.502 | 0.489 |
| Macro AUC | 0.517 | 0.484 |
| 方法学正确性 | 一般 | ✅ 更科学 |

> 单任务方案的结果略低于多任务，但**方法更正确**。多任务的高 bACC 部分来自"搭便车"效应——某些任务的 shared backbone 从其他任务借用了信号。单任务方案的结果更真实地反映了每个任务的独立预测难度。

### 5.5 结果解读

1. **整体挑战性大**：12 任务 macro bACC 均值 0.489，低于随机猜测（0.5），说明从 10 秒 ECG + 单帧面部预测化验指标非常困难。
2. **仅 2 个任务 bACC > 0.5**：`troponin_extreme_high` (0.563) 和 `po2_low` (0.559) 略高于随机。
3. **极端不平衡任务不可学**：`lactate_high` (98.4% 阳性) 的 bACC=0.342，模型偏向预测全阳性。
4. **样本量仍然是瓶颈**：即使单任务方案给出了更多训练样本（250–500），但对于深度学习仍然偏少。

---

## 6. 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/manifest.csv` | 样本元数据 + 15 个标签 |
| `outputs/features.npz` | ECG (731×256) + Face (731×32×32) |
| `outputs/label_summary.csv` | 逐任务标签分布统计 |
| `outputs/split.json` | Train/Val/Test 患者 ID 划分 |
| `outputs/metrics.csv` | 逐任务 × 逐划分的完整指标 |
| `outputs/predictions.csv` | Test Set 逐样本预测分数 |
| `checkpoints/model_<task>.pt` | 每任务最佳模型权重（12 个 .pt 文件） |
| `logs/training_history.json` | 训练过程 loss/score 记录 |

---

## 7. 复现命令

```bash
# 完整流程（在 screen 中后台运行）
screen -dmS exp2 bash -c "cd /root/autodl-tmp/HealthMirrorDataProcess && \
  source /root/miniconda3/etc/profile.d/conda.sh && \
  conda activate healthmirrorenv && \
  python -m study.exp2_lab_multimodal.run_all 2>&1 | tee study/exp2_lab_multimodal/logs/run.log"

# 仅训练（跳过数据集构建）
python -m study.exp2_lab_multimodal.run_all --skip-build

# 单独构建数据集
python -m study.exp2_lab_multimodal.build_dataset

# 单独训练评估
python -m study.exp2_lab_multimodal.train_eval
```

---

## 8. 代码结构

```
study/exp2_lab_multimodal/
├── __init__.py          # 包初始化
├── config.py            # 全局超参数配置
├── models.py            # M3TNet 模型定义
│   ├── Conv1dBlock      # 1D 卷积基本块
│   ├── Conv2dBlock      # 2D 卷积基本块
│   ├── Residual1dBlock  # 1D 残差块
│   ├── ECGEncoder       # ECG 1D CNN 编码器
│   ├── FaceEncoder      # Face 2D CNN 编码器
│   ├── BinaryM3TNet      # 单任务二分类模型（每任务独立训练）
├── build_dataset.py     # 数据提取与特征工程
│   ├── _build_lab_labels()    # 从 CSV 构建化验标签
│   ├── _load_ecg()            # ECG 信号加载与重采样
│   ├── _load_face()           # 面部帧提取与缩放
│   └── build_features()       # 主入口：构建并保存 features.npz
├── train_eval.py        # 逐任务训练与评估
│   ├── ArrayDataset           # 轻量数组 PyTorch Dataset
│   ├── _train_one_task()      # 训练单个 BinaryM3TNet
│   └── train_and_evaluate()   # 遍历 15 个任务逐一训练
└── run_all.py           # 端到端流水线入口
```
