import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical


def get_points_angle(landmark_subset):
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


def input_data(seq, seq_length, model, actions):
    input_data = np.expand_dims(
        np.array(seq[-seq_length:], dtype=np.float32), axis=0)
    y_pred = model.predict(input_data).squeeze()
    i_pred = int(np.argmax(y_pred))
    print(y_pred)

    action = actions[i_pred]
    return action, y_pred


def load_data(path_dir, folder_list):
    data = np.concatenate([
        np.load(path_dir + '/' + folder_list[6]),
        np.load(path_dir + '/' + folder_list[7]),
        np.load(path_dir + '/' + folder_list[8]),
        np.load(path_dir + '/' + folder_list[9]),
        np.load(path_dir + '/' + folder_list[10])
    ], axis=0)
    return data


def load_test_data(path_dir, folder_list):
    data = np.concatenate([
        np.load(path_dir + '/' + folder_list[5]),
        np.load(path_dir + '/' + folder_list[6]),
        np.load(path_dir + '/' + folder_list[7]),
        np.load(path_dir + '/' + folder_list[8]),
        np.load(path_dir + '/' + folder_list[9])
    ], axis=0)
    return data


def seperate_label(data):
    x_data = data[:, :, :-1].astype(np.float32)
    x_labels = data[:, 0, -1]
    y_data = to_categorical(x_labels, num_classes=2)
    y_data = y_data.astype(np.float32)
    return x_data, y_data


def plot_graph(x, y, eval, num):
    plt.subplot(2, 2, num)
    plt.plot(x, y)
    plt.ylabel(eval)
    plt.xlabel('Epochs')