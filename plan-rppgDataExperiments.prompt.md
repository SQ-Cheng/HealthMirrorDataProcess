# Refined Experiment Plan: Pervasive Computing on rPPG/ECG Data

## 1. Status of Existing Experiments & Paradigms

### Identified Failures & Deprecations
- **BP End-to-End DL Overfitting (Exp 01 in note.md):** 1D CNN-LSTM architectures for direct Blood Pressure regression massively overfit. This is because BP labels are missing in 36-66% of the dataset, and the mapping from raw waveforms to BP requires precise vascular compliance modeling that dense neural networks struggle to learn from <1000 noisy samples.
- **Unreliable Manual Feature Extraction:** auto_wash.py implements rigid deterministic algorithms (e.g., peak detection, strict PTT standard deviation thresholding). Because real clinical rPPG is rife with motion artifacts, mathematical peak-finding frequently fails, causing derived features (PTT, HRV, SQI) to be highly unreliable. Thus, traditional ML utilizing these extracted features is not a viable path.
- **Electrical Stimulation:** Deprecated and unreliable. Any autonomic nervous system proxy tests based on this cohort are cancelled.

---

## 2. New Experiment Designs (Fact-Based & Data-Driven)

### Experment A: Cross-Modal Signal Synthesis (1D Bidirectional GAN)
**Objective:** Synthesize high-fidelity contact ECG signals directly from non-contact rPPG facial video signals (and vice versa) using generative networks.
**Necessity & Basis:** 
- *Why:* Ground truth clinical labels (like BP) are scarce, but paired synchronous waveform data (RPPG and ECG channels @ 512Hz) is abundant (~1600 records). We bypass the unreliable manual feature extraction entirely through unsupervised/self-supervised learning. 
ote.md shows initial scaffolding for Exp 02 (Bidirectional GAN) has been started.
- *Methodology:* Implement a 1D CycleGAN or Pix2Pix model. This fits perfectly into the pervasive computing narrative: enabling medical-grade signal (ECG) acquisition via zero-contact sensors (rPPG).

### Experiment B: Deep Learning for Heart Rate & SpO2 Estimation
**Objective:** Predict Heart Rate and Blood Oxygen (SpO2) from raw rPPG waveforms using temporal Convolutional Networks.
**Necessity & Basis:**
- *Why:* BP regressors failed, but exploratory data analysis confirms that heart_rate labels are intact in over 76% of the data, and blood_oxygen in 71%. Unlike BP, heart rate is a fundamental frequency naturally visible in optical volume variations.
- *Methodology:* Use a 1D ResNet with a frequency-domain attention module. By estimating continuous targets that correlate highly with explicit temporal periodicity, the model is significantly less likely to overfit compared to the non-linear morphological mappings required for BP. 

### Experiment C: Data-Driven Deep Artifact Detection (Autoencoder SQI)
**Objective:** Replace the brittle manual data washing algorithms in auto_wash.py with an unsupervised physiological anomaly detector.
**Necessity & Basis:**
- *Why:* The rigid heuristics (abs(p - ptt_median) < 0.1 or bandpass peak rejection) currently fail to gracefully handle true clinical noise. 
- *Methodology:* Train an Autoencoder purely on the top 10% cleanest signal segments (identified by highest SNR power). During inference on the broader dataset, high reconstruction loss serves as a robust, data-driven Signal Quality Index (SQI). This validates a crucial pervasive computing topic: reliable real-world artifact rejection.

---

## 3. Practicability Assessment Based on EDA

- **Generative Pairings (Exp A):** Highly practical. Generative adversarial networks require synchronized unlabelled data, which is essentially the entire 1600-sample dataset, effectively doubling our usable data footprint since we don't have to discard -1 BP targets.
- **HR/SpO2 Targets (Exp B):** Safe & Practical. Shifting the learning objective from BP to variables with much higher representation limits data starvation and exploits physiological signatures explicit in rPPG.
- **Unsupervised Anomaly Detection (Exp C):** Highly practical because it avoids reliance on manual peak-detection packages like neurokit2 or scipy.signal.find_peaks, which break down randomly in real clinical noise scenarios.
