import cv2
import numpy as np
import os
import draw_function
from draw_function import set_cam, custom_landmarks
from operation import get_points_angle, input_data
from keras.models import load_model
from sklearn.preprocessing import Normalizer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
actions = ['standing_side_crunch', 'other']

seq_length = 15
model = load_model(
    'C:/Users/UCL7/VS_kwix/created_dataset/dataset_version9/standing_side_crunch/standing_side_crunch_model_v9_4.h5')
pose = set_cam()

N_scaler = Normalizer()
seq, action_seq = [], []
cap = cv2.VideoCapture(
    'C:/Users/UCL7/VS_kwix/Byungsun_main/create_model_LSTM/side_crunch.mp4')

while True:
    ret, img = cap.read()

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

        if action_seq[-1] == action_seq[-2]:
            if y_pred[0] > 0.8:
                rate = 0
                img = draw_function.drawing_Side_crunch(
                    img, landmark_subset.landmark, rate)
            elif y_pred[0] > 0.6:
                rate = 1
                img = draw_function.drawing_Side_crunch(
                    img, landmark_subset.landmark, rate)
            else:
                rate = 2
                img = draw_function.drawing_Side_crunch(
                    img, landmark_subset.landmark, rate)

    cv2.rectangle(img, (0, 150), (int(y_pred[0]*350), 190), (245, 117, 16), -1)
    cv2.putText(img, 'Standing side crunch',  org=(
        0, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.rectangle(img, (0, 260), (int(y_pred[1]*100), 320), (117, 245, 16), -1)
    cv2.putText(img, 'other',  org=(
        0, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
