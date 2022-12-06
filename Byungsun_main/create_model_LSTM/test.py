import numpy as np
import os
from keras.models import load_model
from operation import load_data_1, load_data_2, seperate_label, reshape_input
from custom_model_61 import LS_61, BiLS_61, CNN_61
from custom_model_99 import LS_99, BiLS_99, CNN_99
from custom_model_109 import LS_109, BiLS_109, CNN_109
from sklearn.metrics import accuracy_score, f1_score


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

for index, name in enumerate(model_name):
    model_v[index].summary()
    print(index)
    for idx, _ in enumerate(_actions):
        path_dir2 = 'C:/Users/UCL7/Desktop/Kwix_HAR/test_dataset_v3/' + _actions[idx]
        folder_list2 = os.listdir(path_dir2)
        test_data = load_data_2(path_dir2, folder_list2)
        test_xdata, test_ydata = seperate_label(test_data)

        print('model name:', name)
        print(f'model shape: {test_xdata.shape}')
        model_path = f'C:/Users/UCL7/Desktop/Kwix_HAR/new_model/v3/{name}_{_actions[idx]}_model.h5'
        model = load_model(model_path)

        if name == 'ConvLSTM':
            test_xdata = reshape_input(test_xdata)

        test_result = model.predict(test_xdata)
        test_result = np.argmax(test_result, axis=1)
        test_y= np.argmax(test_ydata, axis=1)

        acc = accuracy_score(test_y, test_result)
        f1_s = f1_score(test_y, test_result)

        print(f'Action: {_actions[idx]}, Accuracy: {acc}, F1-Score: {f1_s}')
        