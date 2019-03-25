import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from itertools import chain
import gc
import json


PATH = '~/Documents/PythonScripts/Kaggle/PowerLineFault/'
SEED = 42
N_FOLD = 5


def mcc(y_true, y_pred):
    """ Matthews correlation coefficient """

    y_pred = (y_pred > 0.5).astype(int)
    loss = matthews_corrcoef(y_true, y_pred)
    return 'MCC', loss, True


def smote_adataset(x_train, y_train, x_test, y_test):
    """ Oversampling """

    sm = SMOTE(random_state=SEED)
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())
    return x_train_res, pd.Series(y_train_res), x_test, pd.Series(y_test)


def lgbm_model(full_train, y, y_group, best_params):
    """ LightGBM model """

    clfs = []
    folds = StratifiedKFold(n_splits=N_FOLD,
                            shuffle=True,
                            random_state=SEED)

    oof_preds = np.zeros((len(full_train), 2))
    for fold_, (trn_, val_) in enumerate(folds.split(y_group, y_group)):
        trn_ = list(chain(3 * trn_, 3 * trn_ + 1, 3 * trn_ + 2))
        val_ = list(chain(3 * val_, 3 * val_ + 1, 3 * val_ + 2))

        trn_x, trn_y = full_train.iloc[trn_], y.iloc[trn_]
        val_x, val_y = full_train.iloc[val_], y.iloc[val_]

        trn_xa, trn_y, val_xa, val_y = smote_adataset(trn_x.values, trn_y.values, val_x.values, val_y.values)
        trn_x = pd.DataFrame(data=trn_xa, columns=trn_x.columns)
        val_x = pd.DataFrame(data=val_xa, columns=val_x.columns)

        clf = LGBMClassifier(**best_params)
        clf.fit(
            trn_x, trn_y,
            eval_set=[(trn_x, trn_y), (val_x, val_y)],
            eval_metric=mcc,
            verbose=False,
            early_stopping_rounds=20
        )
        clfs.append(clf)

        oof_preds[val_, :] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)
        _, loss, _ = mcc(val_y, oof_preds[val_, 1])
        print('no {}-fold loss: {}'.format(fold_ + 1, loss))

    _, loss, _ = mcc(y_true=y, y_pred=oof_preds[:, 1])
    print('MCC: {:.5f}'.format(loss))

    return clfs, loss, oof_preds


def fit():
    """ Training model """

    full_train = pd.read_csv('../data/train_fft.csv')
    full_train = pd.concat([full_train.iloc[:, :251], full_train.iloc[:, 300:]], axis=1)
    print(full_train.shape)

    meta_train = pd.read_csv(PATH + 'metadata_train.csv')

    groups = meta_train[['id_measurement', 'target']].groupby('id_measurement').sum()
    y_group = groups.target.values

    y = meta_train.target

    with open('lgbm_params.json', 'r') as f:
        best_params = json.load(f)

    clfs, loss, y_pred = lgbm_model(full_train, y, y_group, best_params)

    prev_mcc = 0
    best_thr = 0.5
    for thr in range(10, 100, 5):
        curr_mcc = matthews_corrcoef(y.values, (y_pred[:, 1] >= thr / 100).astype(int))
        if curr_mcc > prev_mcc:
            best_thr = thr / 100
            prev_mcc = curr_mcc

    del full_train, meta_train, y_pred
    gc.collect()

    return clfs, loss, best_thr


def predict(clfs, loss, best_thr):
    """ Making prediction for test dataset """

    full_test = pd.read_csv('../data/test_fft.csv')
    full_test = pd.concat([full_test.iloc[:, :251], full_test.iloc[:, 300:]], axis=1)
    print(full_test.shape)

    meta_test = pd.read_csv(PATH + 'metadata_test.csv')
    signal_id = meta_test.signal_id.values

    preds = None
    for clf in clfs:
        if preds is None:
            preds = clf.predict_proba(full_test, num_iteration=clf.best_iteration_)[:, 1]
        else:
            preds += clf.predict_proba(full_test, num_iteration=clf.best_iteration_)[:, 1]

    preds = preds / len(clfs)
    preds = (preds >= best_thr).astype(int)

    preds_df = pd.DataFrame(preds,
                            columns=['target'])
    preds_df['signal_id'] = signal_id
    preds_df.set_index('signal_id', inplace=True)
    preds_df['target'] = preds_df['target'].astype(int)

    preds_df = preds_df.merge(meta_test, on='signal_id')
    preds_df_groupby = preds_df.groupby('id_measurement').sum()

    for j in [1, 2]:
        ind = preds_df_groupby[preds_df_groupby.target == j].index
        for i in ind:
            preds_df.loc[preds_df.id_measurement == i, 'target'] = j - 1

    preds_df = preds_df[['signal_id', 'target']].set_index('signal_id')
    preds_df.to_csv('../data/lgbm_fft_{}_thr_{}_loss.csv'.format(best_thr, loss))


def main():
    gc.enable()
    clfs, loss, best_thr = fit()
    predict(clfs, loss, best_thr)


if __name__ == '__main__':
    main()
