# Notes
## 2026-02-18
### *Estimation of Beat-by-Beat Blood Pressure and Heart Rate From ECG and PPG Using a Fine-Tuned Deep CNN Model*
- Structure
  ECG - 2-scale conv model1 - 2-scale conv model2 - LSTM1 - output
  PPG - 2-scale conv model3 - 2-scale conv model4 - LSTM2 - output

### *Combined deep CNN–LSTM network-based multitasking learning architecture for noninvasive continuous blood pressure estimation using difference in ECG-PPG features*
- Structure
  ECG-PPG-Difference - 1D conv - batchnorm - Bidirectional LSTM/LSTM - Dense&output

### Advice from LLM
* Gemini: If you want the best results with 1,000 samples, use a Hybrid Approach:
Use a 1D-CNN to extract features from the PPG and ECG.
Concatenate these with handcrafted features (like calculated PTT and Age/Weight if available).
Pass the combined vector into a Random Forest Regressor.
This "Informed Deep Learning" gives you the pattern-recognition of CNNs with the stability of classical ML.

### Experiment 01
- Before:
  * data washing modification: ECG normalize by dividing 32768, no normalization on RPPG - MODIFIED - CHECKING(RPPG norm error?) - DONE
  * data washing on 2 - DONE 
