import cv2
import numpy as np
import time
import os
from draw_function import set_cam, custom_landmarks_13, custom_landmarks_17
from operation import get_points_angle_17, get_points_angle_13
from sklearn.preprocessing import Normalizer


babel_curl = ['babel_curl_F', 'babel_curl_T']
deadlift = ['deadlift_F', 'deadlift_T']
over_head_press = ['over_head_press_F', 'over_head_press_T']
squat = ['squat_F', 'squat_T']
side_raise = ['side_raise_F', 'side_raise_T']

actions = [babel_curl, deadlift, over_head_press, side_raise, squat]
_actions = ['babel_curl', 'deadlift', 'over_head_press', 'side_raise', 'squat']

pose_angles = ['C']
created_time = int(time.time())
sequence_length = 15
N_scaler = Normalizer()
pose = set_cam()

os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/train_dataset_v9/', exist_ok=True)

for i, _ in enumerate(actions):
    os.makedirs('C:/Users/UCL7/Desktop/Kwix_HAR/train_dataset_v9/' + _actions[i], exist_ok=True)
    path_dir = 'E:/' + _actions[i]
    for idx, action in enumerate(_):
        start = time.time()
        data = []
        path1 = path_dir + '/' + _[idx]     # E:/babel_curl/babel_curl_T
        print('action:', action)

        for p in range(1, 9):
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
                    joint = np.zeros((33, 3))
                    for j, lm in enumerate(results.pose_landmarks.landmark):
                        joint[j] = [lm.x, lm.y, lm.z]
                    
                    if idx == 0:
                        label = np.array([0])
                    else: 
                        label = np.array([1])
                        
                    d = np.concatenate([joint.flatten(), label])
                    data.append(d)
                    # label = label.reshape(-1, 1)

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('C:/Users/UCL7/Desktop/Kwix_HAR/train_dataset_v9/' + _actions[i],
                f'raw_{action}_{created_time}'), data)

        full_seq_data = []
        for seq in range(len(data) - sequence_length):
            full_seq_data.append(data[seq:seq + sequence_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('C:/Users/UCL7/Desktop/Kwix_HAR/train_dataset_v9/' + _actions[i],
                f'seq_{action}_{created_time}'), full_seq_data)
        print("Working time : ", time.time() - start)