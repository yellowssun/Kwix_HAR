import cv2
import numpy as np
import time
import os
from draw_function import set_cam
from sklearn.preprocessing import Normalizer
from draw_function import set_cam, custom_landmarks_17
from operation import get_points_angle_17


_actions = ['_side_lunge', '_knee_up', '_leg_raise', '_crunch', '_side_crunch']

pose_angles = ['C']
created_time = int(time.time())
sequence_length = 15
N_scaler = Normalizer()
pose = set_cam()

os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/error_dataset_v3/', exist_ok=True)

for i, _ in enumerate(_actions):
    os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/error_dataset_v3/' + _actions[i], exist_ok=True)
    path_dir = 'C:/Users/UCL7/Desktop/' + _actions[i]
    start = time.time()
    data = []
    label = np.array([0])

    print('action:', _)

    img_file = os.listdir(path_dir)

    for frame in img_file:
        img = path_dir + '/' + frame
        img = cv2.imread(img, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks is not None:
            landmark_subset = custom_landmarks_17(results)
            joint, angle = get_points_angle_17(landmark_subset)

            reshape_angle = angle.reshape(1, -1)
            scaled_angle = N_scaler.fit_transform(reshape_angle)
            scaled_angle = scaled_angle.reshape(-1, 1)

            joint = np.zeros((33, 3))
            for j, lm in enumerate(results.pose_landmarks.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            angle_label = np.array([scaled_angle], dtype=np.float32)
            angle_label = np.append(angle_label, label)

            d = np.concatenate([joint.flatten(), angle_label])
            data.append(d)

    data = np.array(data)
    print(_, data.shape)
    np.save(os.path.join('C:/Users/UCL7/Desktop/Kwix_HAR/error_dataset_v3/' + _actions[i],
            f'raw_{_}_{created_time}'), data)

    full_seq_data = []
    for seq in range(len(data) - sequence_length):
        full_seq_data.append(data[seq:seq + sequence_length])

    full_seq_data = np.array(full_seq_data)
    print(_, full_seq_data.shape)
    np.save(os.path.join('C:/Users/UCL7/Desktop/Kwix_HAR/error_dataset_v3/' + _actions[i],
            f'seq_{_}_{created_time}'), full_seq_data)
    print("Working time : ", time.time() - start)