import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from keras.utils import to_categorical


def get_points_angle_13(landmark_subset):
    """
    13개의 Landmarks, 8개의 각도
    """
    joint = np.zeros((13, 3))
    for j, lm in enumerate(landmark_subset.landmark):
        joint[j] = [lm.x, lm.y, lm.z]

    v1 = joint[[4, 2, 2, 8, 10, 3, 1, 1, 7, 9], :3]
    v2 = joint[[6, 4, 8, 10, 12, 5, 3, 7, 9, 11], :3]
    v = v2 - v1
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt, nt->n',
                                v[[0, 1, 2, 3, 5, 6, 7, 8], :],
                                v[[1, 2, 3, 4, 6, 7, 8, 9], :]))
    angle = np.degrees(angle)
    return joint, angle


def get_points_angle_17(landmark_subset):
    """
    17개의 Landmarks, 10개의 각도
    """
    joint = np.zeros((17, 3))
    for j, lm in enumerate(landmark_subset.landmark):
        joint[j] = [lm.x, lm.y, lm.z]

    v1 = joint[[4, 2, 2, 8, 10, 12, 3, 1, 1, 7, 9, 11], :3]
    v2 = joint[[6, 4, 8, 10, 12, 16, 5, 3, 7, 9, 11, 15], :3]
    v = v2 - v1
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    angle = np.arccos(np.einsum('nt, nt->n',
                                v[[0, 1, 2, 3, 4, 6, 7, 8, 9, 10], :],
                                v[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11], :]))
    angle = np.degrees(angle)
    return joint, angle


def seperate_label(data):
    x_data = data[:, :, :-1].astype(np.float32)
    x_labels = data[:, 0, -1]
    y_data = to_categorical(x_labels, num_classes=3)
    y_data = y_data.astype(np.float32)
    return x_data, y_data


def seperate_angle(data):
    landmark_data = data[:, :, :51].astype(np.float32)
    angle_data = data[:, :, 51:].astype(np.float32)
    print(f'Landmark data shape: {landmark_data.shape}')
    print(f'Angle data shape: {angle_data.shape}')
    return landmark_data, angle_data

    
def input_data(seq, seq_length, model, actions):
    input_data = np.expand_dims(
        np.array(seq[-seq_length:], dtype=np.float32), axis=0)
  
    y_pred = model.predict(input_data).squeeze()
    i_pred = int(np.argmax(y_pred))
    print(y_pred)

    action = actions[i_pred]
    return action, y_pred


def load_data_1(path_dir, folder_list):
    data = np.concatenate([
        np.load(path_dir + '/' + folder_list[2]),
        np.load(path_dir + '/' + folder_list[3]),
        np.load(path_dir + '/' + folder_list[4])
    ], axis=0)
    return data


def load_data_2(path_dir, folder_list):
    data = np.concatenate([
        np.load(path_dir + '/' + folder_list[2]),
        np.load(path_dir + '/' + folder_list[3]),
        np.load(path_dir + '/' + folder_list[4]),
        np.load(path_dir + '/' + folder_list[5]),
        np.load(path_dir + '/' + folder_list[6]),
        np.load(path_dir + '/' + folder_list[7]),
        np.load(path_dir + '/' + folder_list[8]),
        np.load(path_dir + '/' + folder_list[9]),
        np.load(path_dir + '/' + folder_list[10]),
        np.load(path_dir + '/' + folder_list[11]),
        np.load(path_dir + '/' + folder_list[12]),
        np.load(path_dir + '/' + folder_list[13]),
        np.load(path_dir + '/' + folder_list[14]),
        np.load(path_dir + '/' + folder_list[15]),
        np.load(path_dir + '/' + folder_list[16]),
        np.load(path_dir + '/' + folder_list[17]),
        np.load(path_dir + '/' + folder_list[18]),
        np.load(path_dir + '/' + folder_list[19])
    ], axis=0)
    return data


def plot_graph(x, y, eval, num):
    plt.subplot(2, 2, num)
    plt.plot(x, y)
    plt.ylabel(eval)
    plt.xlabel('Epochs')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    
    plt.figure()
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

    print(cm, cm.max)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                    horizontalalignment='center',
                    color='white' if cm[i, j] > thresh else 'black')
    
    
    plt.ylabel('True label')
    plt.xlabel('predicted label')
    plt.tight_layout()