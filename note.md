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
  * `reshape(input, shape)`, `view(shape)`: flatten or un-flatten data without changing the sequence inside the memory.
  * `permute(shape)`, `transpose(input, dim_1, dim_2)`: Like transpose. If your data is `(Time, Batch, Feature)` but your model expects `(Batch, Time, Feature)`, you must use permute`(1, 0, 2)`.
  * `squeeze(input)`: Remove all dimensions that have a size of 1.
  * `nsqueeze(input, dim)`: Add a dimension at the position of `dim`.
  * `cat(tensors, dim=0, *, out=None)`: Concatenate the given tensors in `tensors` in the given EXISTING `dim`.
  * `stack(tensors, dim=0, *, out=None)`: Concatenate the given tensors in `tensors` in the given NEW `dim`.
  * `expand(*sizes)`: Returns a new view of the tensor with singleton dimensions (which size is 1) expanded to a larger size.
  * `repeat(*repeats)`: Repeat the tensor along the dimension: `repeat (torch.Size, int..., tuple of int or list of int)` – The number of times to repeat this tensor along each dimension.

### Experiment 01
- Target:
  * Check the possibility of pure end-to-end deep learning
  * Build a general dataloader
- Before:
  * data washing modification: ECG normalize by dividing 32768, no normalization on RPPG - MODIFIED - CHECKING(RPPG norm error?) - DONE
  * data washing on 2 - DONE, 1 - DONE
  * Understand LSTM
  * Data processing: Data augmentation - sliding windows, 2-sec length, 1-sec step.
- Structure: LSCN
  * Input: ECG: 1024\*1, PPG: 1024\*1.
  * CNN Layer:
    * ECG/PPG: 1024\*1 - 1024\*32 + 1024\*32 - 2-Scale-Conv(kernel=25,9) - 256\*32 + 256\*32 - Max pooling - 2-Scale-Conv(kernel=25,9) - 64\*32 + 64\*32 - stack - 64\*64
  * LSTM Layer: stack.(ECG,PPG) - 64\*128
  * Output Layer: dense - SBP+DBP
- Training:
- Result:
