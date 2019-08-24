
from load_data import *
from model import *
from config import config
import sys

import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    np.random.seed(2019)
    tf.random.set_seed(2019)

    config = config()
    data_file = './sample_data.xlsx'

    if len(sys.argv) > 2:
        data_file = sys.argv[1]

    x_bin_features, feats, tokens, feat_max, y = data_preparation(data_file)

    kf = KFold(n_splits=5, random_state=2019, shuffle=True)
    fold = 1

    accuracy = []
    f1 = []
    auc = []

    for train_index, test_index in kf.split(x_bin_features):

        x_bf_train, x_bf_test = x_bin_features[train_index], x_bin_features[test_index]
        y_train, y_test = y[train_index], y[test_index]


        def train_step(x_bin, t):

            with tf.GradientTape() as tape:
                pred, _, _ = model(x_bin)
                loss = focal_loss(t, pred, alpha=config.alpha, gamma=config.gamma)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_loss(loss)

            return pred


        def test_step(x_bin, t):

            pred, _, _ = model(x_bin)
            loss = focal_loss(t, pred, alpha=config.alpha, gamma=config.gamma)
            test_loss(loss)

            return pred


        model = Graph(tokens, config.embedding, feat_max, config.num_heads, config.dropout_rate)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate, beta_1=config.beta_1,
                                             beta_2=config.beta_2, epsilon=config.epsilon)

        epochs = config.epochs
        batch_size = config.batch_size
        n_batches = x_bin_features.shape[0] // batch_size

        train_loss = tf.keras.metrics.Mean()
        test_loss = tf.keras.metrics.Mean()

        es = []
        preds_temp = []
        stop = False

        for epoch in range(epochs):

            # early stopping
            if stop == False:
                _x_bf_train, _y_train = shuffle(x_bf_train, y_train, random_state=2019)

                for batch in range(n_batches):
                    start = batch * batch_size
                    end = start + batch_size
                    trainpreds = train_step(_x_bf_train[start:end], _y_train[start:end])

                testpreds = test_step(x_bf_test, y_test)
                score = roc_auc_score(y_test, testpreds)
                es.append(score)

                print(' epoch:', epoch, ' auc:', score)
                preds_temp.append(testpreds)

                if len(es) - np.argmax(es) > config.tolerance:
                    stop = True

            else:
                break

        num = np.argmax(es)
        print('fold:', fold, ' epoch:', num)

        pred_temp_thres = np.int32(preds_temp[num] > 0.5)

        acc_temp = accuracy_score(y_test, pred_temp_thres)
        accuracy.append(acc_temp)
        print('fold:', fold, ' accuracy:', acc_temp)

        f1_temp = f1_score(y_test, pred_temp_thres)
        f1.append(f1_temp)
        print('fold:', fold, ' f1_score:', f1_temp)

        auc_temp = roc_auc_score(y_test, preds_temp[num])
        auc.append(auc_temp)
        print('fold:', fold, ' auc:', auc_temp)

        fold += 1

    print('###################################################')
    print('auc:', np.mean(auc))
    print('f1 score:', np.mean(f1))
    print('accuracy:', np.mean(accuracy))
    print('\n')
