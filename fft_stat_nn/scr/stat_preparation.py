import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import gc


PATH = '/Users/vivpavlov/Documents/PythonScripts/Kaggle/PowerLineFault/'
MAX_NUM = 127
MIN_NUM = -128
SAMPLE_SIZE = 800000


def min_max_transf(ts, min_data, max_data, range_needed=(-1, 1)):
    """ Convert data into -1 to 1 """

    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


def transform_ts(ts, n_dim=160):
    ts_std = min_max_transf(ts, min_data=MAX_NUM, max_data=MIN_NUM)
    bucket_size = int(SAMPLE_SIZE / n_dim)
    new_ts = []
    for i in range(0, SAMPLE_SIZE, bucket_size):
        ts_range = ts_std[i:i + bucket_size]
        mean = ts_range.mean()
        std = ts_range.std()
        std_top = mean + std
        std_bot = mean - std
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100])
        max_range = percentil_calc[-1] - percentil_calc[0]
        relative_percentile = percentil_calc - mean
        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range]),
                                      percentil_calc, relative_percentile]))
    return pd.DataFrame(new_ts)


def main():
    gc.enable()
    for label in ['train', 'test']:  # 'train', 'test'
        meta_df = pd.read_csv(PATH + 'metadata_%s.csv' % label)
        id_signal = meta_df.signal_id.unique()

        for col in id_signal:
            df_ = pq.read_table(PATH + '%s.parquet' % label,
                                columns=[str(col)]).to_pandas()

            statistics = transform_ts(df_[str(col)])

            if col == id_signal[0]:
                statistics.to_csv('../data/%s_stat41.csv' % label, header=True, mode='a', index=False)
            else:
                statistics.to_csv('../data/%s_stat41.csv' % label, header=False, mode='a', index=False)

            del df_, statistics
            gc.collect()

            if col % 1000 == 0:
                print(col)

        print('Done')


if __name__ == '__main__':
    main()
