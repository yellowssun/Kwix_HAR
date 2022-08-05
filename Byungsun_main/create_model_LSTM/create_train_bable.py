import cv2
import numpy as np
import time
import os
from draw_function import set_cam, custom_landmarks
from operation import get_points_angle
from sklearn.preprocessing import Normalizer


babel_curl = ['babel_curl_F', 'babel_curl_T']
deadlift = ['deadlift_F', 'deadlift_T']
over_head_press = ['over_head_press_F', 'over_head_press_T']
squat = ['squat_F', 'squat_T']

actions = [squat]
_actions = ['squat']

pose_angles = ['C']
created_time = int(time.time())
sequence_length = 15
N_scaler = Normalizer()

pose = set_cam()

os.makedirs('C:/Users/UCL7/VS_kwix/train_dataset_v3/', exist_ok=True)

for i, _ in enumerate(actions):
    os.makedirs('C:/Users/UCL7/VS_kwix/train_dataset_v3/' + _actions[i], exist_ok=True)
    path_dir = 'E:/' + _actions[i]
    for idx, action in enumerate(_):
        start = time.time()
        data = []
        path1 = path_dir + '/' + _[idx]     # E:/babel_curl/babel_curl_T
        print('action:', action)

        for p in range(1, 7):
            path2 = path1 + '/' + str(p)          # E:/babel_curl/babel_curl_T/1
            print('person:', p)
            print(path2)

            img_file = os.listdir(path2)

            for frame in img_file:
                img = path2 + '/' + frame
                img = cv2.imread(img, cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks is not None:
                    landmark_subset = custom_landmarks(results)
                    joint, angle = get_points_angle(landmark_subset)

                    reshape_angle = np.degrees(angle).reshape(-1, 1)
                    scaled_angle = N_scaler.fit_transform(reshape_angle)

                    angle_label = np.array([scaled_angle], dtype=np.float32)
                    if idx == 0:
                        label = 0   # 불량한 자세
                    else: 
                        label = 1   # 올바른 자세
                    angle_label = np.append(angle_label, label)

                    d = np.concatenate([joint.flatten(), angle_label])
                    data.append(d)


        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('C:/Users/UCL7/VS_kwix/train_dataset_v3/' + _actions[i],
                f'raw_{action}_{created_time}'), data)

        full_seq_data = []
        for seq in range(len(data) - sequence_length):
            full_seq_data.append(data[seq:seq + sequence_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('C:/Users/UCL7/VS_kwix/train_dataset_v3/' + _actions[i],
                f'seq_{action}_{created_time}'), full_seq_data)
        print("Working time : ", time.time() - start)
