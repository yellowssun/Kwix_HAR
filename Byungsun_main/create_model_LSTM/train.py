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
# from custom_model_99 import LS_99, CNN_99, BiLS_99
# from custom_model_61 import LS_61, CNN_61, BiLS_61
from custom_model_109 import LS_109, CNN_109, BiLS_109
# import wandb


# wandb.init(project="17_model")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = '1'

_actions = ['babel_curl', 'knee_up', 'leg_raise', 'overhead_press', 'squat']
loss = 'binary_crossentropy'

model_name1 = 'CNN'
model_name2 = 'LS'
model_name3 = 'BiLS'
model_name = [model_name1, model_name2, model_name3]

# model1 = CNN_99()
# model2 = LS_99()
# model3 = BiLS_99()

# model1 = CNN_61()
# model2 = LS_61()
# model3 = BiLS_61()

model1 = CNN_109()
model2 = LS_109()
model3 = BiLS_109()

model_v = [model1, model2, model3]

os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/new_model/v3', exist_ok=True)
os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v3', exist_ok=True)
os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v3/confusion_matrix', exist_ok=True)
os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v3/loss', exist_ok=True)
os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/js_model/v3', exist_ok=True)
for index, name in enumerate(model_name):
    model_v[index].summary()

    for idx, _ in enumerate(_actions):
        print('action:', _actions[idx])
        path_dir1 = 'C:/Users/UCL7/Desktop/Kwix_HAR/train_dataset_v3/' + _actions[idx]
        folder_list1 = os.listdir(path_dir1)
        train_data = load_data_1(path_dir1, folder_list1)

        path_dir2 = 'C:/Users/UCL7/Desktop/Kwix_HAR/test_dataset_v3/' + _actions[idx]
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
            epochs=100,
            batch_size=64,
            callbacks=[
                ModelCheckpoint(filepath=f'C:/Users/UCL7/Desktop/Kwix_HAR/new_model/v3/{name}_{_actions[idx]}_model.h5',
                                monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto'),
                ReduceLROnPlateau(monitor='val_accuracy', factor=0.2,
                                patience=10, verbose=1, mode='auto')
            ]
        )
        
        model_path = f'C:/Users/UCL7/Desktop/Kwix_HAR/new_model/v3/{name}_{_actions[idx]}_model.h5'
        model = load_model(model_path)



        plt.figure()
        pd.Series(history.history['loss']).plot()
        pd.Series(history.history['val_loss']).plot()

        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.tight_layout()
        plt.legend(['train_loss', 'val_loss'])
        plt.savefig(f"C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v3/loss/{name}_{_actions[idx]}_loss_.png")


        plt.figure()
        pd.Series(history.history['accuracy']).plot()
        pd.Series(history.history['val_accuracy']).plot()
        plt.xlabel("Epoch")
        plt.ylabel("Train Accuracy")
        plt.tight_layout()
        plt.legend(['train_acc', 'val_acc'])
        plt.savefig(f"C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v3/loss/{name}_{_actions[idx]}_acc_.png")

        created_time = int(time.time())
        prediction = model.predict(test_xdata)
        rounded_labels= np.argmax(test_ydata, axis=1)
        rounded_predictions = np.argmax(prediction, axis=1)

        cm = confusion_matrix(y_true=rounded_labels, y_pred=rounded_predictions)
        cm_plot_labels = ['bad', 'good']
        plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion matrix')
        plt.savefig(f'C:/Users/UCL7/Desktop/Kwix_HAR/evaluation_v3/confusion_matrix/{name}_{_actions[idx]}.png')