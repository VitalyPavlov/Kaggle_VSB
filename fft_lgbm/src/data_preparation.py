import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from skimage import util
import gc


PATH = '/Users/vivpavlov/Documents/PythonScripts/Kaggle/PowerLineFault'
IS_SELECTED = False


def fft_all_slices(df_, m=700, step=350):
    slices = util.view_as_windows(df_, window_shape=(m,), step=step)
    win = np.hanning(m + 1)[:-1]
    slices = slices * win
    slices = slices.T
    spectrum = np.fft.fft(slices, axis=0)[:m // 2 + 1:-1]
    spectrum = np.abs(spectrum)
    return pd.DataFrame(np.mean(spectrum, axis=1)).T


def fft_selected_slices(df_, m=700, step=350):
    n = df_.shape[0]
    fd = n / 20e-3
    s = np.fft.rfft(df_)
    freq = np.fft.rfftfreq(n, d=1/fd)
    s[freq > 50] = 0
    trend = np.fft.irfft(s)

    slices = []
    for i in range(0, trend.shape[0] - m, step):
        if trend[i + m] > trend[i]:
            slices.append(df_[i: i + m])
    slices = np.array(slices)

    win = np.hanning(m + 1)[:-1]
    slices = slices * win
    slices = slices.T
    spectrum = np.fft.fft(slices, axis=0)[:m // 2 + 1:-1]
    spectrum = np.abs(spectrum)

    return pd.DataFrame(np.mean(spectrum, axis=1)).T


def get_rest(df_, low, high=None):
    n = df_.shape[0]
    fd = n / 20e-3
    s = np.fft.rfft(df_)
    freq = np.fft.rfftfreq(n, d=1/fd)
    if high:
        s[(freq < low) | (freq > high)] = 0
    else:
        s[freq < low] = 0
    signal = np.fft.irfft(s)
    return pd.DataFrame(signal)


def time_statistics(df_):
    signal_lf = get_rest(df_, low=1e3, high=1e5)
    signal_mf = get_rest(df_, low=1e5, high=1e7)

    d = {
        'std_LF': round(signal_lf.std()[0], 5),
        'skew_LF': round(signal_lf.skew()[0], 5),
        'range_LF': round(signal_lf.max()[0] - signal_lf.min()[0], 5),
        'std_MF': round(signal_mf.std()[0], 5),
        'skew_MF': round(signal_mf.skew()[0], 5),
        'range_MF': round(signal_mf.max()[0] - signal_mf.min()[0], 5)
    }

    return pd.DataFrame(d, index=[0])


def main():
    gc.enable()
    for label in ['train', 'test']:
        meta_df = pd.read_csv(PATH + 'metadata_%s.csv' % label)
        id_signal = meta_df.signal_id.unique()

        for col in id_signal:
            df_ = pq.read_table(PATH + '%s.parquet' % label,
                                columns=[str(col)]).to_pandas()

            time_stats = time_statistics(df_[str(col)].values)

            if IS_SELECTED:
                spectrum = fft_selected_slices(df_[str(col)].values)
            else:
                spectrum = fft_all_slices(df_[str(col)].values)

            res = pd.concat([spectrum, time_stats], axis=1)

            if col == id_signal[0]:
                res.to_csv('../data/%s_fft.csv' % label, header=True, mode='a', index=False)
            else:
                res.to_csv('../data/%s_fft.csv' % label, header=False, mode='a', index=False)

            del df_, spectrum, time_stats, res
            gc.collect()

            if col % 1000 == 0:
                print(col)

        print('Done')


if __name__ == '__main__':
    main()
