import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from skimage import util
import gc


PATH = '/Users/vivpavlov/Documents/PythonScripts/Kaggle/PowerLineFault/'


def fft_all_slices(df_, m=600, step=300):
    slices = util.view_as_windows(df_, window_shape=(m,), step=step)
    win = np.hanning(m + 1)[:-1]
    slices = slices * win
    slices = slices.T
    spectrum = np.fft.fft(slices, axis=0)[:m // 2 + 1:-1]
    spectrum = np.abs(spectrum).T
    spectrum = spectrum.reshape(-1, 65, 298)
    spectrum = spectrum.mean(axis=1)
    return pd.DataFrame(spectrum)


def main():
    gc.enable()
    for label in ['train']:  # 'train', 'test'
        meta_df = pd.read_csv(PATH + 'metadata_%s.csv' % label)
        id_signal = meta_df.signal_id.unique()

        for col in id_signal:
            df_ = pq.read_table(PATH + '%s.parquet' % label,
                                columns=[str(col)]).to_pandas()

            spectrum = fft_all_slices(df_[str(col)].values)

            if col == id_signal[0]:
                spectrum.to_csv('../data/%s_fft.csv' % label, header=True, mode='a', index=False)
            else:
                spectrum.to_csv('../data/%s_fft.csv' % label, header=False, mode='a', index=False)

            del df_, spectrum
            gc.collect()

            if col % 1000 == 0:
                print(col)

        print('Done')


if __name__ == '__main__':
    main()
