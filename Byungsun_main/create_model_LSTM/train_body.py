import numpy as np
import os, time
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import tensorflowjs as tfjs
from sklearn.metrics import confusion_matrix
from keras.metrics import AUC, Precision, Recall
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Bidirectional
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from operation import load_data, load_test_data, seperate_label


figure_num = 1
created_time = int(time.time())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'

actions = ['babel_curl', 'deadlift', 'knee_up', 'leg_raise', 'over_head_press', 'side_crunch', 'side_lunge', 'squart']


os.makedirs('C:/Users/UCL7/VS_kwix/new_model/v2', exist_ok=True)

for idx, _ in enumerate(actions):
    path_dir1 = 'C:/Users/UCL7/VS_kwix/train_dataset_v4/' + actions[idx]
    path_dir2 = 'C:/Users/UCL7/VS_kwix/test_dataset_v4/' + actions[idx]
    folder_list1 = os.listdir(path_dir1)
    folder_list2 = os.listdir(path_dir2)


    data = load_data(path_dir1, folder_list1)
    test_data = load_test_data(path_dir2, folder_list2)
    print(data.shape)
    print(test_data.shape)

    x_data, y_data = seperate_label(data)
    test_xdata, test_ydata = seperate_label(test_data)


    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True,
                    input_shape=x_data.shape[1:3], dropout=0.2)),
        LSTM(128, return_sequences=True, dropout=0.2),
        LSTM(64, dropout=0.2),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(2, activation='softmax')
    ])

    earlystopping = EarlyStopping(monitor='val_loss', patience=10, min_delta=0.0005)

    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy', AUC(), Precision(), Recall()])
    print(model.summary)

    model_path = 'C:/Users/UCL7/VS_kwix/new_model/v4/' + actions[idx] + '_model_v3.2_.h5'

    history = model.fit(
        x_data,
        y_data,
        validation_split=0.2,
        epochs=50,
        callbacks=[
            ModelCheckpoint(model_path,
                            monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                            patience=10, verbose=1, mode='auto')
        ]
    )

    plot_model(model, show_shapes=True)

    model = load_model(model_path)
    tfjs.converters.save_keras_model(model, 'C:/Users/UCL7/VS_kwix/js_model' + '/' + actions[idx] + '_model_tfjs')


# evaluation score
    test_result = model.predict(test_xdata)
    test_result = np.argmax(test_result, axis=1)
    test_y= np.argmax(test_ydata, axis=1)
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    acc = accuracy_score(test_y, test_result)
    precision = precision_score(test_y, test_result)
    recall = recall_score(test_y, test_result)
    print(f'Accuracy: {acc}, Precision: {precision}, Recall: {recall}')


# plot loss and validation loss 
    pd.Series(history.history['loss']).plot(logy=True)
    pd.Series(history.history['val_loss']).plot(logy=True)
    plt.figure(figure_num)
    figure_num += 1
    plt.xlabel("Epoch")
    plt.ylabel("Train Error")
    plt.tight_layout()
    plt.legend(['loss', 'val_loss'])
    plt.savefig(f'C:/Users/UCL7/VS_kwix/train_error_{actions[idx]}_v3.2.png')


# plot confusion matrix
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                        horizontalalignment='center',
                        color='white' if cm[i, j] > thresh else 'black')
        
        
        plt.ylabel('True label')
        plt.xlabel('predicted label')
        plt.tight_layout()
        plt.savefig(f'C:/Users/UCL7/VS_kwix/new_model/v3/confusion_matrix_{actions[idx]}_v3.2.png')

    prediction = model.predict(test_xdata)
    rounded_labels= np.argmax(test_ydata, axis=1)
    rounded_predictions = np.argmax(prediction, axis=1)

    cm = confusion_matrix(y_true=rounded_labels, y_pred=rounded_predictions)
    cm_plot_labels = ['bad', 'good']
    plt.figure(figure_num)
    figure_num += 1
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion matrix')

# --------------------------------------------------------------
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
