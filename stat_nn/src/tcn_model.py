import pandas as pd
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import *
import tensorflow as tf
from keras.layers import *
from bin.tcn_layer import TCN
import gc

np.random.seed(1)
tf.set_random_seed(1)

PATH = '~/Documents/PythonScripts/Kaggle/PowerLineFault/'
SEED = 42
N_FOLD = 5


def mcc_keras(y_true, y_pred):
    """ Matthews correlation coefficient for Keras"""

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = K.eval(mcc_keras(K.variable(y_true, dtype=np.float64),
                                 K.variable((y_proba > threshold), dtype=np.float64)))
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = (best_threshold, best_score)
    return search_result


def lstm_layers(input_shape):
    inp = Input(shape=(input_shape[1], input_shape[2],))

    x = TCN(128, dilations=[1, 2, 4], return_sequences=True, activation='wavenet', name='tnc1')(inp)
    x = TCN(64, dilations=[1, 2, 4], return_sequences=True, activation='wavenet', name='tnc2')(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.5)(x)

    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[mcc_keras])
    return model


def lstm_model(train_stat, y_group):
    folds = StratifiedKFold(n_splits=N_FOLD,
                            shuffle=True,
                            random_state=SEED)

    preds_val = []
    y_val = []
    for fold_, (trn_, val_) in enumerate(folds.split(y_group, y_group)):
        K.clear_session()
        trn_x_stat, trn_y = train_stat[trn_], y_group[trn_]
        val_x_stat, val_y = train_stat[val_], y_group[val_]

        model = lstm_layers(trn_x_stat.shape)
        ckpt = ModelCheckpoint('../data/weights__{}.h5'.format(fold_),
                               save_best_only=True,
                               save_weights_only=True,
                               verbose=1,
                               monitor='val_mcc_keras',
                               mode='max')
        model.fit(trn_x_stat, trn_y,
                  batch_size=128,
                  epochs=50,
                  validation_data=[val_x_stat, val_y],
                  callbacks=[ckpt])
        model.load_weights('../data/weights__{}.h5'.format(fold_))

        preds_val.append(model.predict(val_x_stat, batch_size=512))
        y_val.append(val_y)

    preds_val = np.concatenate(preds_val)[..., 0]
    y_val = np.concatenate(y_val)
    loss = K.eval(mcc_keras(K.variable(y_val.astype(np.float64)),
                            K.variable((preds_val > 0.5).astype(np.float64))))
    print('MCC: {:.5f}'.format(loss))

    return preds_val, y_val, loss, model


def fit():
    """ Training model """

    train_stat = pd.read_csv('../data/train_stat.csv').values
    train_stat = train_stat.reshape(-1, 160, 19)
    train_stat = np.asarray([np.concatenate([train_stat[i],
                                             train_stat[i + 1],
                                             train_stat[i + 2]], axis=1) for i in range(0, len(train_stat), 3)])

    print(train_stat.shape)

    meta_train = pd.read_csv(PATH + 'metadata_train.csv')

    groups = meta_train[['id_measurement', 'target']].groupby('id_measurement').sum()
    y_group = (groups.target > 0).astype(int)

    preds_val, y_val, _, model = lstm_model(train_stat, y_group)
    best_thr, loss = threshold_search(y_val, preds_val)
    print(best_thr, loss)

    del train_stat, meta_train, preds_val, y_val, y_group
    gc.collect()

    return best_thr, loss, model


def predict(best_thr, loss, model):
    """ Making prediction for test dataset """

    test_stat = pd.read_csv('../data/test_stat.csv').values
    test_stat = test_stat.reshape(-1, 160, 19)
    test_stat = np.asarray([np.concatenate([test_stat[i],
                                            test_stat[i + 1],
                                            test_stat[i + 2]], axis=1) for i in range(0, len(test_stat), 3)])

    print(test_stat.shape)

    meta_test = pd.read_csv(PATH + 'metadata_test.csv')
    signal_id = meta_test.signal_id.values

    preds = None
    for fold_ in range(N_FOLD):
        model.load_weights('../data/weights__{}.h5'.format(fold_))
        if preds is None:
            preds = model.predict(test_stat, batch_size=300, verbose=1)
        else:
            preds += model.predict(test_stat, batch_size=300, verbose=1)

    preds = preds / N_FOLD

    preds_test = []
    for pred in preds:
        for i in range(3):
            preds_test.append(pred)

    preds_test = (np.array(preds_test) >= best_thr).astype(int)

    preds_df = pd.DataFrame(preds_test,
                            columns=['target'])
    preds_df['signal_id'] = signal_id
    preds_df.set_index('signal_id', inplace=True)

    preds_df.to_csv('../data/lstm_stat_{}_thr_{}_loss.csv'.format(best_thr, loss))

    del test_stat, meta_test, preds_df
    gc.collect()


def main():
    gc.enable()
    best_thr, loss, model = fit()
    predict(best_thr, loss, model)


if __name__ == '__main__':
    main()
