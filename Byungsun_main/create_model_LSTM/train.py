import numpy as np
import os, time
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import tensorflowjs as tfjs
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
from keras.metrics import AUC, Precision, Recall
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from operation import load_data, load_test_data, seperate_label, plot_confusion_matrix


figure_num = 1
created_time = int(time.time())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'

babel_curl = ['babel_curl_F', 'babel_curl_T']
deadlift = ['deadlift_F', 'deadlift_T']
knee_up = ['knee_up_F', 'knee_up_T']
leg_raise = ['leg_raise_F', 'leg_raise_T']
over_head_press = ['over_head_press_F', 'over_head_press_T']
side_crunch =['side_crunch_F', 'side_crunch_T']
side_lunge = ['side_lunge_F', 'side_lunge_T']
squat = ['squat_F', 'squat_T']


actions = [side_lunge]
_actions = ['side_lunge']
loss = 'binary_crossentropy'


os.makedirs('C:/Users/UCL7/VS_kwix/new_model/v4', exist_ok=True)
os.makedirs('C:/Users/UCL7/VS_kwix/evaluation_v4', exist_ok=True)
os.makedirs('C:/Users/UCL7/VS_kwix/evaluation_v4/confusion_matrix', exist_ok=True)
os.makedirs('C:/Users/UCL7/VS_kwix/evaluation_v4/loss', exist_ok=True)


for idx, action in enumerate(actions):
    print('action:', _actions[idx])
    path_dir1 = 'C:/Users/UCL7/VS_kwix/train_dataset_v4/' + _actions[idx]
    path_dir2 = 'C:/Users/UCL7/VS_kwix/test_dataset_v4/' + _actions[idx]
    folder_list1 = os.listdir(path_dir1)
    folder_list2 = os.listdir(path_dir2)


    data = load_data(path_dir1, folder_list1)
    test_data = load_test_data(path_dir2, folder_list2)

    x_data, y_data = seperate_label(data)
    print(x_data.shape)
    test_xdata, test_ydata = seperate_label(test_data)


    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, batch_size=32,
                    input_shape=x_data.shape[1:3])),
        Bidirectional(LSTM(128, return_sequences=True)),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    earlystopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0005)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy', AUC(), Precision(), Recall()])
    

    model_path = 'C:/Users/UCL7/VS_kwix/new_model' + '/v4/' + _actions[idx] + '_model_v4.0_.h5'

    history = model.fit(
        x_data,
        y_data,
        validation_split=0.1,
        epochs=100,
        callbacks=[
            ModelCheckpoint(model_path,
                            monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                            patience=10, verbose=1, mode='auto')
        ]
    )

    plot_model(model, show_shapes=True)
    model = load_model(model_path)


    test_result = model.predict(test_xdata)
    test_result = np.argmax(test_result, axis=1)
    test_y= np.argmax(test_ydata, axis=1)
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    acc = accuracy_score(test_y, test_result)
    precision = precision_score(test_y, test_result)
    recall = recall_score(test_y, test_result)
    print(f'Accuracy: {acc}, Precision: {precision}, Recall: {recall}')

    # test_results = model.evaluate(
    # test_xdata, test_ydata)

    tfjs.converters.save_keras_model(model, 'C:/Users/UCL7/VS_kwix/js_model' + '/' + _actions[idx] + '_model_tfjs')


    plt.figure()
    pd.Series(history.history['loss']).plot(logy=True)
    pd.Series(history.history['val_loss']).plot(logy=True)
    pd.Series(history.history['accuracy']).plot(logy=True)
    pd.Series(history.history['val_accuracy']).plot(logy=True)
    plt.xlabel("Epoch")
    plt.ylabel("Train Error")
    plt.tight_layout()
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'])
    plt.savefig(f"C:/Users/UCL7/VS_kwix/evaluation_v4/loss/{loss}_{created_time}_{_actions[idx]}_train_error_v5.png")


    # plt.figure()
    # pd.Series(history.history['recall']).plot(logy=True)
    # pd.Series(history.history['val_recall']).plot(logy=True)
    # plt.xlabel("Epoch")
    # plt.ylabel("Recall")
    # plt.tight_layout()
    # plt.legend(['train', 'validation'])
    # plt.savefig(f"C:/Users/UCL7/VS_kwix/new_model/v3/{loss}_{created_time}_{_actions[idx]}_train_recall.png")


    # plt.figure()
    # pd.Series(history.history['precision']).plot(logy=True)
    # pd.Series(history.history['val_precision']).plot(logy=True)
    # plt.xlabel("Epoch")
    # plt.ylabel("Precision")
    # plt.tight_layout()
    # plt.legend(['train', 'validation'])
    # plt.savefig(f"C:/Users/UCL7/VS_kwix/new_model/v3/{loss}_{created_time}_{_actions[idx]}_train_precisinon.png")

    prediction = model.predict(test_xdata)
    rounded_labels= np.argmax(test_ydata, axis=1)
    rounded_predictions = np.argmax(prediction, axis=1)

    cm = confusion_matrix(y_true=rounded_labels, y_pred=rounded_predictions)
    cm_plot_labels = ['bad', 'good']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion matrix')
    plt.savefig(f'C:/Users/UCL7/VS_kwix/evaluation_v4/confusion_matrix/{_actions[idx]}_v5.png')