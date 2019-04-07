# Kaggle_VSB
Kernels for Power Line Fault Detection Competition 

fft_lgbm folder contains 
1) LightGBM with FFT features (CV 0.64, Public LB 0.53, Private LB 0.61)

fft_lstm folder contains 
1) LSTM+Attention with FFT features (CV 0.607, Public LB 0.45, Private LB 0.609)
2) LSTM+Attention with FFT features and group by id_measurements (CV 0.682, Public LB 0.485, Private LB 0.62)

fft_stat_lstm folder contains
1) LSTM+Attention with FFT features and LSTM+Attention with statistics from Bruno Aquino's kernel (CV 0.73, Public LB 0.65, Private LB 0.64)

<img width="398" alt="Screen Shot 2019-04-07 at 11 01 07" src="https://user-images.githubusercontent.com/32665134/55680613-82395180-5924-11e9-8b09-2cd9496840d4.png">
