# Exp2: Multi-Modal Deep Learning for Lab Test Prediction — 实验报告

> **日期**: 2026-07-02  
> **实验目录**: `study/exp2_lab_multimodal/`  
> **环境**: Python 3.12, PyTorch 2.4.1 (CUDA 12.1), Tesla V100 16GB

---

## 1. 实验目标

利用**深度学习多模态模型**，从**10 秒 ECG 信号** + **单帧面部 rPPG 图像**预测患者的**化验指标异常**（15 个二分类任务）。

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

## 3. 模型架构: M3TNet

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│                     M3TNet                          │
│                                                     │
│  ┌──────────────┐    ┌──────────────┐              │
│  │ ECG Encoder  │    │ Face Encoder │              │
│  │  (1D CNN)    │    │  (2D CNN)    │              │
│  │              │    │              │              │
│  │ input:(1,256)│    │ input:(1,32,32)│            │
│  │  ↓           │    │  ↓           │              │
│  │ Conv1d×3     │    │ Conv2d×3     │              │
│  │  + Residual  │    │              │              │
│  │  ↓           │    │  ↓           │              │
│  │ AvgPool      │    │ AvgPool      │              │
│  │  ↓           │    │  ↓           │              │
│  │ FC(64→64)    │    │ FC(32→64)    │              │
│  └──────┬───────┘    └──────┬───────┘              │
│         │  64-d             │  64-d                 │
│         └────────┬──────────┘                      │
│                  ↓ Concat (128-d)                   │
│           ┌──────────────┐                          │
│           │ Fusion MLP   │                          │
│           │ FC(128→64)   │                          │
│           │ + SiLU + BN  │                          │
│           │ + Dropout 0.5│                          │
│           └──────┬───────┘                          │
│                  ↓ (64-d shared)                    │
│     ┌────────┬───┴───┬─────────────┐               │
│     │ Head 1 │ Head 2 │ ... │ Head K│              │
│     │ FC(1)  │ FC(1)  │     │ FC(1) │              │
│     │ Sigmoid│ Sigmoid│     │Sigmoid│              │
│     └────────┴────────┴─────┴───────┘               │
│           K = 12 binary outputs                     │
└─────────────────────────────────────────────────────┘
```

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
| Fusion MLP + 12 Heads | ~8K |
| **总计** | **64,388** |

---

## 4. 训练配置

### 4.1 数据划分

采用**患者级别分组划分**（按 `hospital_id`），确保同一患者的所有样本只出现在一个划分中：

| 划分 | 患者数 | 样本数 | 占比 |
|------|:-----:|:-----:|:---:|
| Train | 68 | 468 | ~64% |
| Validation | 16 | 125 | ~17% |
| Test | 21 | 138 | ~19% |

### 4.2 超参数

| 参数 | 值 | 说明 |
|------|:--:|------|
| Batch Size | 32 | |
| Learning Rate | 3×10⁻⁴ | 初始学习率 |
| LR Scheduler | ReduceLROnPlateau | mode=max, factor=0.5, patience=15 |
| Optimizer | AdamW | weight_decay=1×10⁻³ |
| Max Epochs | 200 | |
| Early Stopping | patience=40 | 基于 validation balanced accuracy |
| Dropout | 0.5 | Fusion MLP 中 |
| Loss | BCEWithLogitsLoss | reduction='none', pos_weight=2.0 |
| Gradient Clipping | max_norm=1.0 | |

### 4.3 损失函数

多任务掩码二元交叉熵：

$$\mathcal{L} = \frac{1}{\sum m_{i,t}} \sum_{i=1}^{B} \sum_{t=1}^{K} m_{i,t} \cdot \ell(y_{i,t}, \hat{y}_{i,t})$$

其中 $m_{i,t} \in \{0, 1\}$ 为标签有效性掩码（缺失标签不计入损失），$\ell$ 为带正类权重的 BCEWithLogitsLoss。

### 4.4 任务过滤

训练前自动检测训练集中各类别样本数，跳过任一类别样本不足 8 个的任务：

| 跳过的任务 | 原因 |
|-----------|------|
| `troponin_high` | 0 个阴性样本（全部阳性） |
| `hemoglobin_low` | 仅 5 个阴性样本 |
| `coronary_context` | 0 个阳性样本（全部阴性） |

最终训练 **12 个有效任务**。

---

## 5. 实验结果

### 5.1 训练曲线

| 指标 | 值 |
|------|:--:|
| 最佳 Epoch | 38 |
| 总训练 Epoch | 78 (early stopped) |
| 最终 Train Loss | 0.470 |
| 最终 Val Loss | 0.850 |
| 最佳 Val Balanced Accuracy | 0.547 |

### 5.2 Test Set 整体指标（12 任务 Macro 平均）

| 指标 | 均值 ± 标准差 |
|------|:-----------:|
| Balanced Accuracy | 0.502 ± 0.048 |
| ROC-AUC | 0.517 ± 0.165 |
| F1 Score | 0.554 ± 0.285 |
| Average Precision | 0.517 ± 0.254 |

### 5.3 Test Set 逐任务结果

| 任务 | bACC | ROC-AUC | F1 | n | 阳性率 |
|------|:---:|:------:|:--:|:--:|:-----:|
| lactate_high | 0.500 | 0.960 | 0.993 | 138 | 98.6% |
| glucose_high | 0.490 | 0.396 | 0.821 | 138 | 71.0% |
| po2_low | 0.484 | 0.632 | 0.824 | 138 | 73.9% |
| pco2_abnormal | 0.500 | 0.345 | 0.934 | 138 | 87.7% |
| **high_blood_pressure** | **0.570** | **0.570** | 0.523 | 93 | 37.6% |
| lactate_moderate_high | 0.445 | 0.396 | 0.095 | 138 | 31.2% |
| troponin_extreme_high | 0.516 | 0.513 | 0.538 | 138 | 48.6% |
| glucose_marked_high | 0.461 | 0.437 | 0.554 | 138 | 44.9% |
| hemoglobin_moderate_low | 0.530 | 0.538 | 0.367 | 138 | 24.6% |
| **po2_moderate_low** | **0.561** | **0.572** | 0.440 | 138 | 36.2% |
| **pco2_low** | **0.554** | 0.460 | 0.269 | 138 | 17.4% |
| pco2_high | 0.411 | 0.389 | 0.291 | 138 | 39.9% |

### 5.4 结果解读

1. **整体模型未学到强信号**：12 任务的 macro balanced accuracy 均值仅 0.502，接近随机猜测（0.5）。
2. **个别任务略高于随机**：`high_blood_pressure` (bACC=0.570)、`po2_moderate_low` (0.561)、`pco2_low` (0.554) 略高于 0.5，但统计意义上不显著。
3. **高阳性率任务虚高**：`lactate_high` (98.6% 阳性) 的 ROC-AUC=0.960 是假象——模型只需预测全阳性即可获得高 AUC，但 bACC 仍为 0.5。
4. **过拟合明显**：train loss 从 1.09 降至 0.47，val loss 仅从 1.00 降至 0.85，gap 持续扩大。

### 5.5 局限性分析

| 因素 | 影响 |
|------|------|
| **信号-标签弱关联** | 10 秒 ECG + 单帧面部图像承载的信息量有限，难以反映全身化验指标 |
| **患者级划分严格** | 训练/测试患者完全不重叠，模型无法利用患者特异性特征 |
| **样本量有限** | 468 训练样本 ÷ 12 任务，每个任务平均仅 39 样本 |
| **标签噪音** | 化验指标可能来自不同时间点，与 ECG/面部采集时刻不一定同步 |
| **类别不平衡** | 多个任务阳性率 >85% 或 <15%，bACC 受拖累 |

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
| `checkpoints/best_model.pt` | 最佳模型权重 |
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
│   └── M3TNet           # 多模态多任务融合网络
├── build_dataset.py     # 数据提取与特征工程
│   ├── _build_lab_labels()    # 从 CSV 构建化验标签
│   ├── _load_ecg()            # ECG 信号加载与重采样
│   ├── _load_face()           # 面部帧提取与缩放
│   └── build_features()       # 主入口：构建并保存 features.npz
├── train_eval.py        # 训练与评估流水线
│   ├── MultiModalDataset      # PyTorch Dataset
│   ├── train_epoch()          # 单 epoch 训练
│   ├── evaluate()             # 模型评估
│   ├── _filter_active_tasks() # 自动任务过滤
│   └── train_and_evaluate()   # 完整训练+评估主函数
└── run_all.py           # 端到端流水线入口
```
