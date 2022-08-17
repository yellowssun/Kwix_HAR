import numpy as np
import os, time
import pandas as pd
import matplotlib.pyplot as plt
import tensorflowjs as tfjs
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.metrics import AUC, Precision, Recall
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from operation import load_data, load_test_data, seperate_label, plot_confusion_matrix, seperate_angle
from custom_model import CNN, BiLS, BiLS_CNN, CNN_BiLS, LS
# import wandb
from sklearn.model_selection import train_test_split


# wandb.init(project="17_model")
created_time = int(time.time())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'

_actions = ['side_raise', 'side_crunch', 'side_lunge', 'babel_curl', 'deadlift', 'knee_up', 'leg_raise', 'over_head_press', 'squat']
# _actions = ['side_raise', 'side_crunch', 'side_lunge']

loss = 'binary_crossentropy'

# model_name1 = 'BiLS_CNN'
model_name3 = 'LS'
model_name2 = 'BiLS'
# model_name4 = 'two_input_model'
model_name = [ model_name2, model_name3]

# model1 = BiLS_CNN()
model3 = LS()
model2 = BiLS()
# model4 = two_input_model()
model_v = [model2, model3]

os.makedirs('C:/Users/dimli/Desktop/Kwix_HAR/new_model/v6.3', exist_ok=True)
os.makedirs('C:/Users/dimli/Desktop/Kwix_HAR/evaluation_v6.3', exist_ok=True)
os.makedirs('C:/Users/dimli/Desktop/Kwix_HAR/evaluation_v6.3/confusion_matrix', exist_ok=True)
os.makedirs('C:/Users/dimli/Desktop/Kwix_HAR/evaluation_v6.3/loss', exist_ok=True)
os.makedirs('C:/Users/dimli/Desktop/Kwix_HAR/js_model/v6.3', exist_ok=True)
for index, name in enumerate(model_name):
    model_v[index].summary()
    for idx, _ in enumerate(_actions):
        print('action:', _actions[idx])
        path_dir1 = 'C:/Users/dimli/Desktop/Kwix_HAR/train_dataset_v7/' + _actions[idx]
        folder_list1 = os.listdir(path_dir1)
        data = load_data(path_dir1, folder_list1)

        train_data, test_data = train_test_split(data, test_size=0.1, train_size=0.9, random_state=2022)

        x_data, y_data = seperate_label(train_data)
        test_xdata, test_ydata = seperate_label(test_data)

        print('train data shape:', x_data.shape[1:3])
        print('test data shape:', test_xdata.shape[1:3])
        

        model = model_v[index]
        print('model name:', name)

        earlystopping = EarlyStopping(monitor='val_loss', patience=20, min_delta=0.0001, mode='min', restore_best_weights=True)
        model.compile(optimizer='adam', loss='binary_crossentropy',
                    metrics=['accuracy', AUC(), Precision(), Recall()])
        

        history = model.fit(
            x_data,
            y_data,
            validation_split=0.135,
            epochs=100,
            batch_size=32,
            callbacks=[
                ModelCheckpoint(filepath=f'C:/Users/dimli/Desktop/Kwix_HAR/new_model/v6.3/{name}_{_actions[idx]}_model.h5',
                                monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
                ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                                patience=10, verbose=1, mode='auto')
            ]
        )
        
        model_path = f'C:/Users/dimli/Desktop/Kwix_HAR/new_model/v6.3/{name}_{_actions[idx]}_model.h5'
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

        tfjs.converters.save_keras_model(model, f'C:/Users/dimli/Desktop/Kwix_HAR/js_model/v6.3/{name}_{_actions[idx]}_model_tfjs')


        plt.figure()
        pd.Series(history.history['loss']).plot()
        pd.Series(history.history['val_loss']).plot()

        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.tight_layout()
        plt.legend(['train_loss', 'val_loss'])
        plt.savefig(f"C:/Users/dimli/Desktop/Kwix_HAR/evaluation_v6.3/loss/{name}_{_actions[idx]}_loss_.png")


        plt.figure()
        pd.Series(history.history['accuracy']).plot()
        pd.Series(history.history['val_accuracy']).plot()
        plt.xlabel("Epoch")
        plt.ylabel("Train Accuracy")
        plt.tight_layout()
        plt.legend(['train_acc', 'val_acc'])
        plt.savefig(f"C:/Users/dimli/Desktop/Kwix_HAR/evaluation_v6.3/loss/{name}_{_actions[idx]}_acc_.png")

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
        plt.savefig(f'C:/Users/dimli/Desktop/Kwix_HAR/evaluation_v6.3/confusion_matrix/{name}_{_actions[idx]}.png')