import numpy as np
import os, time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflowjs as tfjs
from keras.utils import plot_model
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.metrics import AUC, Precision, Recall
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from operation import load_data, load_test_data, seperate_label, plot_confusion_matrix, seperate_angle
from custom_model import CNN, BiLS, BiLS_CNN, CNN_BiLS, LS, two_input_BiLS, two_input_CNN



created_time = int(time.time())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'

_actions = ['babel_curl', 'deadlift', 'knee_up', 'leg_raise', 'over_head_press', 'side_raise', 'side_crunch', 'side_lunge', 'squat']
# _actions = ['side_raise', 'side_crunch', 'side_lunge']
model_name1 = 'two_input_BiLS'
model_name2 = 'two_input_CNN'
loss = 'binary_crossentropy'

os.makedirs('C:/Users/dimli/Desktop/Kwix_HAR/new_model/v6.1', exist_ok=True)
os.makedirs('C:/Users/dimli/Desktop/Kwix_HAR/evaluation_v6.1', exist_ok=True)
os.makedirs('C:/Users/dimli/Desktop/Kwix_HAR/evaluation_v6.1/confusion_matrix', exist_ok=True)
os.makedirs('C:/Users/dimli/Desktop/Kwix_HAR/evaluation_v6.1/loss', exist_ok=True)
os.makedirs('C:/Users/dimli/Desktop/Kwix_HAR/js_model/v6.1', exist_ok=True)
for idx, _ in enumerate(_actions):
    print('action:', _actions[idx])
    path_dir1 = 'C:/Users/dimli/Desktop/Kwix_HAR/train_dataset_v6/' + _actions[idx]
    path_dir2 = 'C:/Users/dimli/Desktop/Kwix_HAR/test_dataset_v6/' + _actions[idx]
    folder_list1 = os.listdir(path_dir1)
    folder_list2 = os.listdir(path_dir2)


    data = load_data(path_dir1, folder_list1)
    test_data = load_test_data(path_dir2, folder_list2)

    x_data, y_data = seperate_label(data)
    test_xdata, test_ydata = seperate_label(test_data)

    print('input data shape:', x_data.shape[1:3])
    train_landmark, train_angle = seperate_angle(x_data)
    test_landmark, test_angle = seperate_angle(test_xdata)


    model = two_input_BiLS()
    model.summary()
    plot_model(model=model, to_file=f'{_actions[idx]}_model_v6.png', show_shapes=True)

    earlystopping = EarlyStopping(monitor='val_loss', patience=30, min_delta=0.0001, mode='auto')
    model.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy', AUC(), Precision(), Recall()])
    


    history = model.fit(
        [train_landmark, train_angle],
        y_data,
        validation_split=0.3,
        epochs=100,
        batch_size=32,
        callbacks=[
            ModelCheckpoint(filepath=f'C:/Users/dimli/Desktop/Kwix_HAR/new_model/v6.1/{model_name}_{_actions[idx]}_model.h5',
                            monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                            patience=10, verbose=1, mode='auto')
        ]
    )
    model_path = f'C:/Users/dimli/Desktop/Kwix_HAR/new_model/v6.1/{model_name}_{_actions[idx]}_model.h5'
    model = load_model(model_path)

    test_result = model.predict([test_landmark, test_angle])
    test_result = np.argmax(test_result, axis=1)
    test_y= np.argmax(test_ydata, axis=1)
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    acc = accuracy_score(test_y, test_result)
    precision = precision_score(test_y, test_result)
    recall = recall_score(test_y, test_result)
    print(f'Accuracy: {acc}, Precision: {precision}, Recall: {recall}')

    # test_results = model.evaluate(
    # test_xdata, test_ydata)

    tfjs.converters.save_keras_model(model, f'C:/Users/dimli/Desktop/Kwix_HAR/js_model/v6.1/{model_name}_{_actions[idx]}_model_tfjs')


    plt.figure()
    pd.Series(history.history['loss']).plot()
    pd.Series(history.history['val_loss']).plot()
    pd.Series(history.history['accuracy']).plot()
    pd.Series(history.history['val_accuracy']).plot()
    plt.xlabel("Epoch")
    plt.ylabel("Train Error")
    plt.tight_layout()
    plt.legend(['train_loss', 'val_loss', 'train_acc', 'val_acc'])
    plt.savefig(f"C:/Users/dimli/Desktop/Kwix_HAR/evaluation_v6.1/loss/{model_name}_{_actions[idx]}_train_error_.png")


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

    prediction = model.predict([test_landmark, test_angle])
    rounded_labels= np.argmax(test_ydata, axis=1)
    rounded_predictions = np.argmax(prediction, axis=1)

    cm = confusion_matrix(y_true=rounded_labels, y_pred=rounded_predictions)
    cm_plot_labels = ['bad', 'good']
    plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion matrix')
    plt.savefig(f'C:/Users/dimli/Desktop/Kwix_HAR/evaluation_v6.1/confusion_matrix/{model_name}_{_actions[idx]}.png')