# Notes

## Overall
- Data quality
  * Mirror1 Final Update: 20251009, Patient_ID > 315.
  * **TODO**
    * Check data auto wash process - whether all-0 values are excepted

## 2026-02-18
### *Estimation of Beat-by-Beat Blood Pressure and Heart Rate From ECG and PPG Using a Fine-Tuned Deep CNN Model*
- Structure
  ECG - 2-scale conv model1 - 2-scale conv model2 - LSTM1 - output
  PPG - 2-scale conv model3 - 2-scale conv model4 - LSTM2 - output

### *Combined deep CNN–LSTM network-based multitasking learning architecture for noninvasive continuous blood pressure estimation using difference in ECG-PPG features*
- Structure
  ECG-PPG-Difference - 1D conv - batchnorm - **Bidirectional** LSTM/LSTM - Dense&output

### Advice from LLM
* Gemini: If you want the best results with 1,000 samples, use a Hybrid Approach:
Use a 1D-CNN to extract features from the PPG and ECG.
Concatenate these with handcrafted features (like calculated PTT and Age/Weight if available).
Pass the combined vector into a Random Forest Regressor.
This "Informed Deep Learning" gives you the pattern-recognition of CNNs with the stability of classical ML.

### Learn LSTM
[Guide](https://medium.com/analytics-vidhya/lstms-explained-a-complete-technically-accurate-conceptual-guide-with-keras-2a650327e8f2)
- Preprocess 
  * data structure: 3d tensor 
    * N: samples
    * T: time steps, that's to say, datapoints in one piece of sample, time*acq freq
    * F: features, or number of signals. e.g. 2 for PPG+ECG.
  * data feeding
    * Z-score norm, perform on dataloader. Datawasher output raw signal, only resampled.
    * 
- Process of working: unroll gradually
  * LSTM unroll over time steps; on each step, the hidden state ("memory") is updated. We ignore the output of all previous steps and only take which of the final step.
- Model structure
  * input: T*F, 2D tensor
  * LSTM layer 1 (what's LSTM layer?) passes sequences to the next layer.
  * LSTM layer 2 collapses (or, discard the output (hidden state) of previous steps and only keep which of the last step) time dimension and only return the final summary.
  * Dense - output.
  * IN THE LSTM LAYER:
    * input data $x_t$
    * the Cell State $C_t$: Carry the information and updated between time steps.
    * the Hidden State $H_t$
    * the Forget gate $F_t$, the Input gate $I_t$, the Output gate $O_t$, all defined by $H_{t-1}$ and $x_{t-1}$. e.g. $F_t=\sigma(W_f\cdot [H_{t-1}, x_t]+b_f)$
    * LSTM CELL: $C_t = C_{t-1}\times F_t+I_t\times \bar{C_t}$, in which $\bar{C_t}=tanh(W_c\cdot [H_{t-1}, x_t]+b_c), $H_t = tanh(C_{t})\times O_t$
- LSTM in PyTorch
  * ` torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)`
    * `input_size`: The number of input features. For example, CNN encoded ECG signal may have 16/32/64 features.
    * `hidden_size`: The number of features in the hidden state. For example, the hidden size of 128 means the LSTM could remember 128 features.
    * `batch_first`: For the shape of input/output tensors. `True`-`(batch, seq, feature)`, `False`-`(seq, batch, feature)`.
    * Inputs: `input, (h_0, c_0)`
      * `input`: `(seq_length, input_size)` for unbatched, `(batch_size, seq_length, input_size)` for batched, `batch_first=True`.
      * `h_0`: `(num_layers, hidden_size)` for unbatched, `(num_layers, batch_size, hidden_size)` for batched.
      * `c_0`: `(num_layers, hidden_size)` for unbatched, `(num_layers, batch_size, hidden_size)` for batched.
    * Outputs: `output, (h_n, c_n)`
      * `output`: `(seq_length, hidden_size)` for unbatched, `(batch_size, seq_length, hidden_size)` for batched, `batch_first=True`.

## 2026-02-19
### Learn PyTorch
- Tensor shape
  * `reshape(tensor, shape)`, `view(shape)`: flatten or un-flatten data without changing the sequence inside the memory.
  * `permute(shape)`, `transpose(input, dim_1, dim_2)`: Like transpose. If your data is `(Time, Batch, Feature)` but your model expects `(Batch, Time, Feature)`, you must use permute`(1, 0, 2)`.
  * `squeeze(input)`: Remove all dimensions that have a size of 1.
  * `nsqueeze(input, dim)`: Add a dimension at the position of `dim`.
  * `cat(tensors, dim=0, *, out=None)`: Concatenate the given tensors in `tensors` in the given EXISTING `dim`.
  * `stack(tensors, dim=0, *, out=None)`: Concatenate the given tensors in `tensors` in the given NEW `dim`.
  * `expand(*sizes)`: Returns a new view of the tensor with singleton dimensions (which size is 1) expanded to a larger size.
  * `repeat(*repeats)`: Repeat the tensor along the dimension: `repeat (torch.Size, int..., tuple of int or list of int)` – The number of times to repeat this tensor along each dimension.
- Tensor modification
  * `chunk(tensor, chunks, dim)`: Divide a certain tensor into `chunks` seperate parts equally.
  * `split(tensor, *sizes, dim)`: `*sizes=32` divide into equal sizes (32), `*sizes=[16,32,48]` divide into different sizes.
  * manual slicing like `[0:mid], [mid:-1]`.

## 2026-02-21
### Experiment 01
- Target:
  * Check the possibility of pure end-to-end deep learning
  * Build a general dataloader
- Before:
  * Z-score normalization on both ECG and RPPG.
  * data washing on 2 - DONE, 1 - DONE
  * Understand LSTM
  * Data processing: Data augmentation - sliding windows, 3-sec length, 1-sec step.
- Structure: LSCN
  * Input: ECG: 1024\*1, PPG: 1024\*1.
  * CNN Layer:
    * ECG/PPG: 1024\*1 - 1024\*32 + 1024\*32 - 2-Scale-Conv(kernel=25,9) - 256\*32 + 256\*32 - Max pooling - 2-Scale-Conv(kernel=25,9) - 64\*32 + 64\*32 - stack - 64\*64
  * LSTM Layer: stack.(ECG,PPG) - 64\*128
  * Output Layer: dense - SBP+DBP
- Training & Result:
  * Exp 01-01
    * Under hyperparameters:
      ```python
      BATCH_SIZE = 32
      LEARNING_RATE = 1e-4
      EPOCHS = 50
      VAL_RATIO = 0.2
      SEED = 42
      WINDOW_SEC = 3.0
      STEP_SEC = 1.0
      TARGET_LENGTH = 1024
      ```
    * The best result is:
      ```python
      SBPTr -0.06+-14.731, DBPTr -0.01+-10.411, SBPVa 0.05+-14.393, DBPVa -0.71+-10.474
      ```
    * Problems:
      OVERFITTING?

  * Exp 01-02
    * Model Modification:
      Add 0.2 dropout in the end of CNN module
    * Hyperparameters same as 01-01
    * The best result is:
      ```python
      SBPTr -0.85+-19.575, DBPTr -0.06+-10.714, SBPVa -1.72+-16.610, DBPVa -0.28+-10.426
      ```
    * Problems:
      Looks like 01-01 has overfitted
      OVERFITTING?

  * Exp 01-03
    * Model Modification:
      Changed 0.3 dropout in the end of CNN module and before the output layer
    * Hyperparameter modification:
      EPOCHS = 100
    * The best result is:
      ```python
      SBPTr -0.05+-14.670, DBPTr -0.09+-10.453, SBPVa -0.43+-14.098, DBPVa 0.50+-10.354
      ```
    * Problems:
      OVERFITTING?

  * Exp 01-04
    * Model Modification:
      Changed 0.5 dropout in the end of CNN module and before the output layer
    * Dataset modification:
      Orthogonal dataset: samples with the same hospital patient id are put into either tr/va set
    * Hyperparameters same as 01-03

    * IN-TRAINING PROBLEM:
      Different rand seed creates confusing situation: The SD of either Tr or Va set is exteremely high. **CHECK FOR ANY FAULTY SAMPLE.**
      Significant waving of training/validating loss. **Too high dropout?** Training/validating loss goes in the different direction.
      Happens in the first about 50 epoches.

    * Result:
      The best one on va set is not the best on tr set.
      Near-end overall best result:
      ```python
      SBPTr -1.02+-20.171, DBPTr -0.00+-9.904, SBPVa -6.08+-17.786, DBPVa -6.01+-12.200
      ```

## 2026-02-22
**TODO**
* fix model structure: redundant maxpool? - FIXED, cat/stack - FIXED
  * Further model structure fixing after exp.
* re-train under 0.3 dropout - TRAINING - OVERFITTED
* check for any new models published - CHECKING
* check for faulty data (maybe not exist, just caused by too large dropout) - FAUTY DATA EXISTS - CHECKING
* **Download resources for working without network!!!** - EXPIRED
### Experiment 01
* Exp 01-05
  * Model Structure fix:
    * Redundant maxpool deleted
    * Downsampling by conv1d stride=2
    * parameter count: 169,346
    * dropout changed to 0.3, 0.3 for conv, output
  * Hyperparameters:
    ```
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 100
    VAL_RATIO = 0.2
    SEED = 45
    WINDOW_SEC = 3.0
    STEP_SEC = 1.0
    TARGET_LENGTH = 1024
    ```
  * Result:
    * Very likely overfitted - waving loss on validation set
    * no useful data.

* Exp 01-06
  * Model structure:
    * LSTM hidden size changed to 64.
    * 103,170 parameters
  * Result
    * Likely overfitted, as 01-05. No improvement on validation set after the 4th epoch.
    * 91st epoch: -1.47    22.276    -0.06     9.899  |   -5.45    18.543    -6.22    12.190

* Exp 01-07
  * Model structure:
    * Optimized feature extraction. No cross-feature conv1d.
    * LSTM input size changed to 128 respectively. hidden size remain 64.
    * 189,442 parameters
  * Result:
    * Likely overfitted.
    * 98th epoth: -0.59    19.860    -0.20     9.971  |   -4.93    18.419    -6.03    12.281

* Exp 01-08
  * Model structure:
    * Due to different sampling freq compared with the reference, the kernel sizes were changed to 49 and 15.
    * 312,322 parameters
  * Result
    * Very likely overfitted. Ref using ~38,000 parameters.
    * Likely a larger model works better - achieved new validation best on 47th epoch,
    * Overall -4+-18 SBP, -6+-12 DBP over validation set. `SEED=45`
    * Near AAMI accuracy on training set however signiificant SBP error on validation set. `SEED=46`
    * Like `SEED=45` for `SEED=47`
    * Looks like faulty data in tr set: 45,47,48,49; va set: 42,46
  * **TODO**
    * Check for faulty data
    * Learn CNN+Transformer structure

## 2026-02-23
### *A paralleled CNN and Transformer network for PPG-based non-invasive blood pressure estimation*
- Model Structure
  * PCTN: CNN+Transformer
  * CNN Block: Resnet/UNet style. Pyramid CNN with U-Net style???
  * Transformer block: 
  * Data preprocess: 125Hz 1024pts - ~8s

### Different networks for different purposes in medical signal processing:
- UNet - Reconstruction
- ResNet - Classification
- CNN (+LSTM/Transformer) - Prediction
- For ECG+PPG->BP prediction, newer approaches include:
  * Mamba-UNet: the bottleneck of UNet replaced by a "selective state space" layer
  * Diffusion models
  * Self-supervised foundation models - might a worse and more costy approach

### Learn transformer

### Learn CBAM
- Parts: Channel-wise attention, spatial attention.
- Channel-wise attention:
  * Extracted features channel_num\*feature_shape(h\*w or length) - Parallel(AvgPool, MaxPool) on feature axis(channel_cnt\*1\*1) - 3-layer-MLP - Element-wise summation - M_c
  * AvgPool and MaxPool results share the same MLP and summed up.
  * Pooling operations size - One channel into 1 num.
  * So the output size of the pooling module is channel_cnt \* 1 \* 1, respectively the input size of the raw feature map is channel_cnt \* h \* w.
- Spatial attention:
  * Extracted features channel_num\*feature_shape(h\*w or length) - Parallel(AvgPool, MaxPool) on channel axis(1\*feature_shape) - Concatenation - Conv - M_s
- Sequential arrangement: channel-wise - spatial

### Learn PCTN
* Overall Structure
  * Stem module: Preprocess, include 1d conv, batch norm, max pooling.
  * Stem output - Conv block: like U-Net, concat lower/higher dimension features - Spatial attention
  * Stem output - 3\*Transformers block - Spatial attention
  * Conv/Trans concat - Channel attention

**TODO**
- Calculate the mean/stddev of bp in the collected dataset
- Understand PCTN structure
- Design EXP2 which use PCTN/Mamba-UNet/Diffusion models. learn them respectively.

## 2026-02-24
### Experiment 01
* Exp 01-09
  * Fixed error data for testing purpose
  * In-training performance was equal to guessing the average.
  
* Exp 01-10
  * Model modification:
    * Changed to kernel size 25,9, dropout=0.5
    * 189,442 params
    * MSE Loss, 100 Epochs
  * Result
    * Likely overfitted on training data
    * -0.40    11.306    -0.14     8.545  |   -2.07    16.933    -0.01    13.026 (Learning rate 1e-4)
    * -0.02     9.493     0.03     7.745  |   -3.51    17.459    -0.88    13.117 (Learning rate 1e-3)
    * ```
      [BP Stats]                SBP Mean    SBP SD    DBP Mean    DBP SD
                  ----------  ----------  --------  ----------  --------
                      Train      116.34     14.40       70.91     10.27
                        Val      121.57     14.82       72.87     11.78
      ```

* Exp 01-11
  * Model modification:
    * Input size reduced to 512. Channel count reduced to 16.
    * 47,618 params
    * Huber loss, 100 epochs
  * Overfitted
    * 0.08     8.670    -0.00     7.613  |   -5.55    17.515    -1.63    13.037

## 2026-02-25
### Experiment 01
* Exp 01-12
  * Training modification
    * Window length reduced to 2 sec
  * 
