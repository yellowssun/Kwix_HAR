import numpy as np
import os, time
import pandas as pd
import matplotlib.pyplot as plt
# import tensorflowjs as tfjs
from sklearn.metrics import confusion_matrix
from keras.optimizers import Adam
from keras.models import load_model
from keras.metrics import Precision, Recall
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from operation import load_data_1, load_data_2, seperate_label, plot_confusion_matrix
# from custom_model_61 import LS, CNN, BiLS
from custom_model_99 import LS, CNN, BiLS
from sklearn.model_selection import train_test_split
# import wandb


# wandb.init(project="17_model")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'

_actions = ['babel_curl', 'deadlift', 'leg_raise', 'overhead_press', 'side_crunch', 'squat']
# 'babel_curl', 'deadlift', 'deadlift', 'knee_up', 'leg_raise', 'overhead_press', 'side_crunch', 'side_lunge', 'side_raise', 'squat'
loss = 'binary_crossentropy'

model_name1 = 'CNN'
model_name2 = 'LS'
model_name3 = 'BiLS'
model_name = [model_name1, model_name2, model_name3]

model1 = CNN()
model2 = LS()
model3 = BiLS()
model_v = [model1, model2, model3]

os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/new_model/v1', exist_ok=True)
os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v1', exist_ok=True)
os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v1/confusion_matrix', exist_ok=True)
os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v1/loss', exist_ok=True)
os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/js_model/v1', exist_ok=True)
for index, name in enumerate(model_name):
    model_v[index].summary()
    print('123123123')

    for idx, _ in enumerate(_actions):
        if _actions[idx] == 'knee_up':
            path_dir1 = 'C:/Users/UCL7/Desktop/Kwix_HAR/train_dataset_v1/' + _actions[idx]
            folder_list1 = os.listdir(path_dir1)
            data = load_data_1(path_dir1, folder_list1)

            train_data, test_data = train_test_split(data, test_size=0.1, train_size=0.9, random_state=1998)
            x_data, y_data = seperate_label(train_data)
            test_xdata, test_ydata = seperate_label(test_data)
            print(x_data.shape)

        else:
            print('action:', _actions[idx])
            path_dir1 = 'C:/Users/UCL7/Desktop/Kwix_HAR/train_dataset_v1/' + _actions[idx]
            folder_list1 = os.listdir(path_dir1)
            train_data = load_data_1(path_dir1, folder_list1)

            path_dir2 = 'C:/Users/UCL7/Desktop/Kwix_HAR/test_dataset_v1/' + _actions[idx]
            folder_list2 = os.listdir(path_dir2)
            test_data = load_data_2(path_dir2, folder_list2)

            x_data, y_data = seperate_label(train_data)
            test_xdata, test_ydata = seperate_label(test_data)
            print(f'train data shape: {x_data.shape}, test data shape: {test_data.shape}')

        model = model_v[index]
        print('model name:', name)

        earlystopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001, mode='min', restore_best_weights=True)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                    metrics=['accuracy', Precision(), Recall()])

        history = model.fit(
            x_data,
            y_data,
            validation_split=0.2,
            epochs=75,
            batch_size=32,
            callbacks=[
                ModelCheckpoint(filepath=f'C:/Users/UCL7/Desktop/Kwix_HAR/new_model/v1/{name}_{_actions[idx]}_model.h5',
                                monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
                ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                                patience=10, verbose=1, mode='auto')
            ]
        )
        
        model_path = f'C:/Users/UCL7/Desktop/Kwix_HAR/new_model/v1/{name}_{_actions[idx]}_model.h5'
        model = load_model(model_path)

        # test_results = model.evaluate(
        # test_xdata, test_ydata)

        # tfjs.converters.save_keras_model(model, f'C:/Users/UCL7/Desktop/Kwix_HAR/js_model/v1/{name}_{_actions[idx]}_model_tfjs')


        plt.figure()
        pd.Series(history.history['loss']).plot()
        pd.Series(history.history['val_loss']).plot()

        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.tight_layout()
        plt.legend(['train_loss', 'val_loss'])
        plt.savefig(f"C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v1/loss/{name}_{_actions[idx]}_loss_.png")


        plt.figure()
        pd.Series(history.history['accuracy']).plot()
        pd.Series(history.history['val_accuracy']).plot()
        plt.xlabel("Epoch")
        plt.ylabel("Train Accuracy")
        plt.tight_layout()
        plt.legend(['train_acc', 'val_acc'])
        plt.savefig(f"C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v1/loss/{name}_{_actions[idx]}_acc_.png")

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

        created_time = int(time.time())
        prediction = model.predict(test_xdata)
        rounded_labels= np.argmax(test_ydata, axis=1)
        rounded_predictions = np.argmax(prediction, axis=1)

        cm = confusion_matrix(y_true=rounded_labels, y_pred=rounded_predictions)
        cm_plot_labels = ['bad', 'good']
        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion matrix')
        plt.savefig(f'C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v1/confusion_matrix/{name}_{_actions[idx]}.png')