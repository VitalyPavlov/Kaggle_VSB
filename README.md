# Kaggle_VSB
Kernels for Power Line Fault Detection Competition 

**fft_lgbm** folder contains
1) LightGBM with FFT features (CV 0.64, Public LB 0.53, Private LB 0.61)

**fft_nn** folder contains 
1) LSTM+Attention with FFT features where each phase is considered separately (CV 0.607, Public LB 0.45, Private LB 0.609)

2) LSTM+Attention with FFT features (CV 0.682, Public LB 0.485, Private LB 0.62)

3) TCN with FFT features (CV 0.71, Public LB 0.52, Private LB 0.61)

**fft_stat_nn** folder contains
1) LSTM+Attention for FFT features and LSTM+Attention for time-series statistics from Bruno Aquino's kernel (CV 0.73, Public LB 0.65, Private LB 0.64)

**stat_nn** folder contains
1) TCN model with time-series statistics from Bruno Aquino's kernel (CV 0.74, Public LB 0.61, Private LB 0.67)
