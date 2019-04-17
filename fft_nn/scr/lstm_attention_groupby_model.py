import pandas as pd
from keras.models import Model
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import *
import tensorflow as tf
from keras.layers import *
from bin.attention_layer import Attention
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

    x = Bidirectional(LSTM(41, return_sequences=True), merge_mode='concat')(inp)
    x = Bidirectional(LSTM(41, return_sequences=True), merge_mode='concat')(x)
    x = Attention(input_shape[1])(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[mcc_keras])
    return model


def lstm_model(full_train, y_group):
    folds = StratifiedKFold(n_splits=N_FOLD,
                            shuffle=True,
                            random_state=SEED)

    preds_val = []
    y_val = []
    for fold_, (trn_, val_) in enumerate(folds.split(y_group, y_group)):
        K.clear_session()
        trn_x, trn_y = full_train[trn_], y_group[trn_]
        val_x, val_y = full_train[val_], y_group[val_]

        model = lstm_layers(trn_x.shape)
        ckpt = ModelCheckpoint('../data/weights__{}.h5'.format(fold_),
                               save_best_only=True,
                               save_weights_only=True,
                               verbose=1,
                               monitor='val_mcc_keras',
                               mode='max')
        model.fit(trn_x, trn_y,
                  batch_size=128,
                  epochs=50,
                  validation_data=[val_x, val_y],
                  callbacks=[ckpt])
        model.load_weights('../data/weights__{}.h5'.format(fold_))

        preds_val.append(model.predict(val_x, batch_size=512))
        y_val.append(val_y)

    preds_val = np.concatenate(preds_val)[..., 0]
    y_val = np.concatenate(y_val)
    loss = K.eval(mcc_keras(K.variable(y_val.astype(np.float64)),
                                K.variable((preds_val > 0.5).astype(np.float64))))
    print('MCC: {:.5f}'.format(loss))

    return preds_val, y_val, loss, model


def fit():
    """ Training model """

    full_train = pd.read_csv('../data/train_fft.csv').values
    full_train = full_train.reshape(-1, 41, 298)
    full_train = np.asarray([np.concatenate([full_train[i],
                                             full_train[i + 1],
                                             full_train[i + 2]], axis=1) for i in range(0, len(full_train), 3)])
    print(full_train.shape)

    meta_train = pd.read_csv(PATH + 'metadata_train.csv')

    groups = meta_train[['id_measurement', 'target']].groupby('id_measurement').sum()
    y_group = (groups.target > 1).astype(int)

    preds_val, y_val, _, model = lstm_model(full_train, y_group)
    best_thr, loss = threshold_search(y_val, preds_val)
    print(best_thr, loss)

    del full_train, meta_train, preds_val, y_val, y_group
    gc.collect()

    return best_thr, loss, model


def predict(best_thr, loss, model):
    """ Making prediction for test dataset """

    full_test = pd.read_csv('../data/test_fft.csv').values
    full_test = full_test.reshape(-1, 41, 298)
    full_test = np.asarray([np.concatenate([full_test[i],
                                            full_test[i + 1],
                                            full_test[i + 2]], axis=1) for i in range(0, len(full_test), 3)])
    print(full_test.shape)

    meta_test = pd.read_csv(PATH + 'metadata_test.csv')
    signal_id = meta_test.signal_id.values

    preds = None
    for fold_ in range(N_FOLD):
        model.load_weights('../data/weights__{}.h5'.format(fold_))
        if preds is None:
            preds = model.predict(full_test, batch_size=300, verbose=1)
        else:
            preds += model.predict(full_test, batch_size=300, verbose=1)

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
    preds_df.to_csv('../data/lstm_fft_{}_thr_{}_loss.csv'.format(best_thr, loss))

    del full_test, meta_test, preds_df
    gc.collect()


def main():
    gc.enable()
    best_thr, loss, model = fit()
    predict(best_thr, loss, model)


if __name__ == '__main__':
    main()
