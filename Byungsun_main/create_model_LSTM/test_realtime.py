import cv2
import numpy as np
import os
import draw_function
from draw_function import set_cam, custom_landmarks
from operation import get_points_angle, input_data
from keras.models import load_model
from sklearn.preprocessing import Normalizer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

actions = ['crunch', 'lying_leg_raise',
           'side_lunge', 'standing_knee_up',
           'standing_side_crunch']


seq_length = 15
model = load_model('C:/Users/UCL7/VS_kwix/created_dataset/dataset_version8/model_v8.h5')


pose = set_cam()
seq = []
action_seq = []
N_scaler = Normalizer()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()
    
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks is not None:
        landmark_subset = custom_landmarks(results)
        joint, angle = get_points_angle(landmark_subset)
        reshape_angle = np.degrees(angle).reshape(-1, 1)
        scaled_angle = N_scaler.fit_transform(reshape_angle).flatten()
        
        d = np.concatenate([joint.flatten(), scaled_angle])
        seq.append(d)

        if len(seq) < seq_length:
            continue

        action, y_pred = input_data(seq, seq_length, model, actions)
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action
            rate = 0
            if action == 'crunch':
                img = draw_function.drawing_Crunch(img, landmark_subset.landmark, rate)
            elif action == 'lying_leg_raise':
                img = draw_function.drawing_Leg_raise(img, landmark_subset.landmark, rate)
            elif action == 'side_lunge':
                img = draw_function.drawing_Side_lunge(img, landmark_subset.landmark, rate)
            elif action == 'standing_knee_up':
                img = draw_function.drawing_Knee_up(img, landmark_subset.landmark, rate)
            elif action == 'standing_side_crunch':
                img = draw_function.drawing_Side_crunch(img, landmark_subset.landmark, rate)
            else:
                img = draw_function.drawing_All(img, landmark_subset.landmark)

        cv2.putText(img, f'{this_action.upper()}', org=(0, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(250, 190, 50), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break