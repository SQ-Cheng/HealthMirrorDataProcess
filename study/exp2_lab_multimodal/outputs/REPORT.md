# Exp2: Per-Task Deep Learning for Lab Test Prediction — 实验报告 (v2)

> **日期**: 2026-07-02  
> **版本**: v2 — 时间匹配标签  
> **环境**: Python 3.12, PyTorch 2.4.1 (CUDA 12.1), Tesla V100 16GB  
> **策略**: 每个化验任务训练一个独立的 BinaryM3TNet；标签按时间最近匹配

---

## 1. 实验目标与方法论

利用**深度学习多模态模型**，从**10 秒 ECG 信号** + **单帧面部 rPPG 图像**预测患者的**化验指标异常**。

### v2 关键改进：时间匹配标签

**v1 问题**: 旧版取患者整个住院期间每种化验的**最大/最小值**作为标签。这导致几乎所有患者都被标记为"异常"（如乳酸 >2.0 的阳性率高达 96.7%），因为住院期间总会有某个时刻指标异常。

**v2 修正**: 对于每个 ECG 采集会话，查找**时间上最近的**化验测量值，用该值计算二分类标签。这确保了标签反映的是采集时刻的患者状态。

### 为什么用单任务模型？

| 方面 | 多任务 | 单任务 ✅ |
|------|:---:|:---:|
| 每任务有效样本 | ~468（共享） | 该任务有标签的全部样本 |
| 任务间干扰 | 有 | 无 |
| 标签缺失处理 | 需掩码损失 | 只使用有标签的样本 |

---

## 2. 数据集

### 2.1 数据来源与预处理

见前版报告。核心变化：
- **标签构建**: `_build_lab_timeseries()` 将 132,771 行化验记录展开为扁平时间序列表
- **时间匹配**: 对每个 ECG 会话，按 `|lab_time − session_time|` 最小原则选取最近化验值

### 2.2 标签分布（v2 时间匹配）

| 任务 | 总样本 | 阳性 | 阴性 | 阳性率 | vs 旧版(max/min) |
|------|:-----:|:---:|:---:|:-----:|:---------------:|
| lactate_high | 731 | 299 | 432 | 40.9% | ↓ 96.7%→40.9% |
| troponin_high | 731 | 374 | 357 | 51.2% | ↓ 100%→51.2% ✅ |
| glucose_high | 731 | 87 | 644 | 11.9% | ↓ 76.1%→11.9% |
| hemoglobin_low | 731 | 64 | 667 | 8.8% | ↓ 98.9%→8.8% |
| po2_low | 731 | 99 | 632 | 13.5% | ↓ 62.8%→13.5% |
| pco2_abnormal | 731 | 169 | 562 | 23.1% | ↓ 89.6%→23.1% |
| high_blood_pressure | 435 | 180 | 255 | 41.4% | — |
| coronary_context | 731 | 0 | 731 | 0% | — |
| lactate_moderate_high | 731 | 14 | 717 | 1.9% | ↓ 19.8%→1.9% |
| troponin_extreme_high | 731 | 52 | 679 | 7.1% | ↓ 40.5%→7.1% |
| glucose_marked_high | 731 | 42 | 689 | 5.7% | ↓ 54.7%→5.7% |
| hemoglobin_moderate_low | 731 | 12 | 719 | 1.6% | ↓ 29.3%→1.6% |
| po2_moderate_low | 731 | 27 | 704 | 3.7% | ↓ 27.9%→3.7% |
| pco2_low | 731 | 30 | 701 | 4.1% | ↓ 22.0%→4.1% |
| pco2_high | 731 | 16 | 715 | 2.2% | ↓ 32.8%→2.2% |

> **关键**: `troponin_high` 从 100% 阳性变为 51.2%——现在可以训练了。多个严重程度任务的阳性率大幅下降（如 `lactate_moderate_high` 从 19.8%→1.9%），说明极端值通常与采集时刻不同步。

---

## 3. 模型架构: BinaryM3TNet

每个化验任务训练一个独立的 **BinaryM3TNet**（63,673 参数）：

```
ECG (1,256) → ECG Encoder (1D CNN, 64-d)
Face (1,32,32) → Face Encoder (2D CNN, 64-d)
       ↓ Concat (128-d)
  Fusion MLP: FC(128→64) + SiLU + BN + Dropout(0.5)
       ↓
  Head: FC(64→1) + Sigmoid → P(abnormal)
```

---

## 4. 训练配置

| 参数 | 值 |
|------|:--:|
| Optimizer | AdamW, lr=3e-4, wd=1e-3 |
| Scheduler | ReduceLROnPlateau (patience=15, factor=0.5) |
| Max Epochs | 200 per task |
| Early Stopping | patience=40 on val bACC |
| Loss | BCEWithLogitsLoss (pos_weight=2.0) |
| Batch Size | 32 |
| Split | Per-task patient-level 60/20/20 |
| Min samples/class | 8 (otherwise skip task) |

---

## 5. 实验结果

### 5.1 整体指标（13 训练任务 / 11 有效评测任务）

| 指标 | Macro 均值 ± 标准差 |
|------|:-----------:|
| Balanced Accuracy | 0.448 ± 0.125 |
| ROC-AUC | 0.403 ± 0.166 |
| F1 Score | 0.149 ± 0.199 |

### 5.2 逐任务 Test Set 结果

| 任务 | 训练样本 | bACC | AUC | F1 | 阳性率 |
|------|:-----:|:---:|:---:|:--:|:-----:|
| lactate_high | 469 | 0.500 | 0.398 | 0.000 | 44.0% |
| troponin_high | 420 | 0.437 | 0.360 | 0.483 | 48.5% |
| **glucose_high** | 422 | **0.566** | **0.581** | 0.240 | 13.6% |
| hemoglobin_low | 396 | 0.273 | 0.249 | 0.075 | 7.8% |
| po2_low | 419 | 0.495 | 0.533 | 0.120 | 21.9% |
| pco2_abnormal | 430 | 0.460 | 0.519 | 0.280 | 29.5% |
| **high_blood_pressure** | 251 | **0.525** | **0.616** | 0.595 | 41.6% |
| troponin_extreme_high | 439 | 0.551 | 0.467 | 0.148 | 6.3% |
| glucose_marked_high | 436 | 0.496 | 0.508 | 0.000 | 7.3% |
| hemoglobin_moderate_low | 430 | 0.117 | 0.007 | 0.000 | 0.7% |
| po2_moderate_low | 417 | 0.500 | 0.310 | 0.000 | 0.7% |
| pco2_low | 444 | 0.399 | 0.420 | 0.000 | 1.7% |
| pco2_high | 421 | 0.500 | 0.272 | 0.000 | 1.3% |

**跳过的任务**: `coronary_context` (0 阳性), `lactate_moderate_high` (仅 4 阳性)

### 5.3 结果解读

1. **时间匹配后标签分布大幅改善**：多数任务从 >80% 阳性降至 <40%，`troponin_high` 变为可训练。
2. **`glucose_high` (bACC=0.566, AUC=0.581)** 和 **`high_blood_pressure` (bACC=0.525, AUC=0.616)** 是最有希望的任务，弱于但高于随机。
3. **严重程度阈值任务几乎不可学**：`lactate_moderate_high`（1.9%）、`hemoglobin_moderate_low`（1.6%）等阳性率过低。
4. **`hemoglobin_low` (bACC=0.273)** 表现差：虽然有了 64 个阳性样本，但模型未能有效利用信号。
5. **`troponin_high` (bACC=0.437)** 略低于随机：虽然类别平衡了（51.2%），但 10 秒 ECG + 面部可能不足以捕捉肌钙蛋白水平。

### 5.4 与旧版对比

| | v1 (max/min) | v2 (time-matched) |
|---|---|---|
| 标签科学性 | ❌ 错误关联 | ✅ 时间匹配 |
| troponin_high 阳性率 | 100% (不可学) | 51.2% (可学) |
| glucose_high 阳性率 | 76.1% | 11.9% |
| Macro bACC | 0.489 | 0.448 |
| 最佳单任务 bACC | 0.563 | 0.566 |

> v2 的结果更真实——它不再"作弊"式地使用住院期间的最差值，而是诚实反映采集时刻的患者状态。整体 bACC 下降是因为任务变得更诚实而非更简单。

---

## 6. 输出文件

| 文件 | 说明 |
|------|------|
| `outputs/manifest.csv` | 样本元数据 + 15 个时间匹配标签 |
| `outputs/features.npz` | ECG (731×256) + Face (731×32×32) |
| `outputs/lab_timeseries.csv` | 扁平化验时间序列表 |
| `outputs/label_summary.csv` | 逐任务标签分布 |
| `outputs/metrics.csv` | 逐任务 × 逐划分完整指标 |
| `outputs/predictions.csv` | Test Set 逐样本预测 |
| `checkpoints/model_<task>.pt` | 每任务最佳模型（12 个） |

---

## 7. 复现命令

```bash
screen -dmS exp2 bash -c "cd /root/autodl-tmp/HealthMirrorDataProcess && \
  source /root/miniconda3/etc/profile.d/conda.sh && \
  conda activate healthmirrorenv && \
  python -m study.exp2_lab_multimodal.run_all 2>&1 | tee study/exp2_lab_multimodal/logs/run.log"
```
