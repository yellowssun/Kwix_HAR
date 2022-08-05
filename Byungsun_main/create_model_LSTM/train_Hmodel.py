import numpy as np
import os
import matplotlib.pyplot as plt
import tensorboard
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from keras.metrics import AUC, Precision, Recall
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from operation import load_data, seperate_label, plot_graph




"""Hyperparameter tuning
    Learning rate
    Loss Function
    Traing loop
    Hidden Unit = []
"""


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([64, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_EARLYSTOP = hp.HParam('earlystop', hp.RealInterval(0.0001, 0.0005))

METRIC_ACCURACY = 'accuracy'
METRIC_AUC = 'auc'
METRIC_PRECISION = 'precision'
METRIC_RECALL = 'recall'


with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_EARLYSTOP],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy'),
                 hp.Metric(METRIC_AUC, display_name='Auc'),
                 hp.Metric(METRIC_PRECISION, display_name='Precision'),
                 hp.Metric(METRIC_RECALL, display_name='Recall')],
    )

actions = ['crunch', 'lying_leg_raise', 'side_lunge',
           'standing_knee_up', 'standing_side_crunch']

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'
os.makedirs('new_model', exist_ok=True)


def hparam_model(hparams, x_data, y_data, test_xdata, test_ydata):
    model = Sequential([
        Bidirectional(LSTM(hparams[HP_NUM_UNITS], return_sequences=True,
                           input_shape=x_data.shape[1:3], dropout=hparams[HP_DROPOUT])),
        LSTM(128, return_sequences=True, dropout=hparams[HP_DROPOUT]),
        LSTM(64, dropout=hparams[HP_DROPOUT]),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy', AUC(), Precision(), Recall()])

    earlystopping = EarlyStopping(
        monitor='val_loss', patience=5, min_delta=hparams[HP_EARLYSTOP])

    model.fit(
        x_data,
        y_data,
        validation_split=0.1,
        epochs=100,
        callbacks=[earlystopping,
                   ModelCheckpoint('C:/Users/UCL7/VS_kwix/created_dataset/new_model/' + str(action) + '_nmodel_' + str(session_num) + '.h5',
                                   monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
                   ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                                     patience=10, verbose=1, mode='auto')
                   ])

    print({h.name: hparams[h] for h in hparams})

    test_results = model.evaluate(test_xdata, test_ydata)

    return test_results


def run(run_dir, hparams, x_data, y_data, test_xdata, test_ydata):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        test_results = hparam_model(hparams, x_data, y_data, test_xdata, test_ydata)
        accuracy = test_results[1]
        auc = test_results[2]
        precision = test_results[3]
        recall = test_results[4]
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)
        tf.summary.scalar(METRIC_AUC, auc, step=2)
        tf.summary.scalar(METRIC_PRECISION, precision, step=3)
        tf.summary.scalar(METRIC_RECALL, recall, step=4)


session_num = 0
for action in actions:
    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for earlystop_rate in (HP_EARLYSTOP.domain.min_value, HP_EARLYSTOP.domain.max_value):
                path_dir1 = 'C:/Users/UCL7/VS_kwix/created_dataset/dataset_version9' + '/' + str(action)
                path_dir2 = 'C:/Users/UCL7/VS_kwix/test_dataset' + '/' + str(action)
                folder_list1 = os.listdir(path_dir1)
                folder_list2 = os.listdir(path_dir2)

                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                    HP_EARLYSTOP: earlystop_rate
                }

                data = load_data(path_dir1, folder_list1)
                test_data = load_data(path_dir2, folder_list2)

                x_data, y_data = seperate_label(data)
                test_xdata, test_ydata = seperate_label(test_data)


                run_name = "run-%d" % session_num
                print('--- Starting trial: %d', run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning/' + run_name, hparams, x_data, y_data, test_xdata, test_ydata)
                session_num +=1

                # y_val_loss = history.history['val_loss']
                # y_loss = history.history['loss']
                # x_len = np.arange(len(y_loss))
                # y_test_loss = np.full(x_len.shape, test_results[0])
                # y_test_acc = np.full(x_len.shape, test_results[1])
                # y_test_auc = np.full(x_len.shape, test_results[2])
                # y_test_precision = np.full(x_len.shape, test_results[3])
                # y_test_recall = np.full(x_len.shape, test_results[4])

                # plt.figure(figsize = (9, 8))
                # plot_graph(x_len, y_loss, 'Loss', 1)
                # plot_graph(x_len, y_val_loss, 'Val_loss', 2)

                # plt.figure(figsize = (9, 8))
                # plot_graph(x_len, y_test_acc, 'ACC', 1)
                # plot_graph(x_len, y_test_auc, 'AUC', 2)
                # plot_graph(x_len, y_test_precision, 'PRECISION', 3)
                # plot_graph(x_len, y_test_recall, 'RECALL', 4)
                # plt.show()