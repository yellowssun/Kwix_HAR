import cv2
import numpy as np
import os
import draw_function
from draw_function import set_cam, custom_landmarks_17
from operation import get_points_angle_17, input_data
from keras.models import load_model
from sklearn.preprocessing import Normalizer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
actions = ['bad', 'good', 'other']
seq_length = 15
model_1 = load_model(
    'C:/Users/UCL7/Desktop/Kwix_HAR/new_model/v9/RNN_babel_curl_model.h5')
model_2 = load_model(
    'C:/Users/UCL7/Desktop/Kwix_HAR/new_model/v9/LS_babel_curl_model.h5')
model_3 = load_model(
    'C:/Users/UCL7/Desktop/Kwix_HAR/new_model/v9/CNN_babel_curl_model.h5')

pose = set_cam()
seq = []
action_seq = []
N_scaler = Normalizer()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
count = 0
state = 0

while cap.isOpened():
    ret, img = cap.read()
    img0 = img.copy()
    
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks is not None:
        landmark_subset = custom_landmarks_17(results)
        joint, angle = get_points_angle_17(landmark_subset)
        reshape_angle = angle.reshape(1, -1)
        scaled_angle = N_scaler.fit_transform(reshape_angle)
        scaled_angle = scaled_angle.reshape(-1, 1).flatten()

        
        d = np.concatenate([joint.flatten(), scaled_angle])
        seq.append(d)

        if len(seq) < seq_length:
            continue

        action, y_pred = input_data(seq, seq_length, model_1, actions)
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        if action_seq[-1] == action_seq[-2]:
            if y_pred[1] > 0.8:
                rate = 0
                img = draw_function.drawing_Side_crunch(
                    img, landmark_subset.landmark, rate)
            elif y_pred[0] > 0.8:
                rate = 1
                img = draw_function.drawing_Side_crunch(
                    img, landmark_subset.landmark, rate)

            else:
                rate = 2
                img = draw_function.drawing_Side_crunch(
                    img, landmark_subset.landmark, rate)

        print(f'Left knee angle: {angle[3]}')
        print(f'Right knee angle: {angle[9]}')

        if state == 0:
            if angle[0] < 30:
                state = 1

        else:
            if angle[0] > 60:
                count += 1  
                state = 0


    cv2.rectangle(img, (0, 150), (int(y_pred[0]*100), 190), (245, 117, 16), -1)
    cv2.putText(img, 'Bad',  org=(
        0, 180), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.rectangle(img, (0, 260), (int(y_pred[1]*100), 320), (117, 245, 16), -1)
    cv2.putText(img, 'Good',  org=(
        0, 300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.putText(img, f'count: {count}',  org=(
        0, 420), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break



    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break