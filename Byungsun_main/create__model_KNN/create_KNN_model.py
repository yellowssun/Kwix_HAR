import os
import cv2
import numpy as np
import draw_function_KNN
from draw_function_KNN import set_cam
from sklearn.neighbors import KNeighborsClassifier
from operation_KNN import load_data, load_test_data, seperate_label, get_points_angle, input_data

path_dir1 = 'C:/Users/UCL7/VS_kwix/created_dataset/dataset_version9/standing_side_crunch'
path_dir2 = 'C:/Users/UCL7/VS_kwix/test_dataset/standing_side_crunch'
folder_list1 = os.listdir(path_dir1)
folder_list2 = os.listdir(path_dir2)


data = load_data(path_dir1, folder_list1)
test_data = load_test_data(path_dir2, folder_list2)

x_data, y_data = seperate_label(data)
test_xdata, test_ydata = seperate_label(test_data)

x_data = x_data.reshape(6305, 915)
test_xdata = test_xdata.reshape(3897, 915)

classifier =  KNeighborsClassifier(n_neighbors=5)

classifier.fit(x_data, y_data)

seq_length = 15
seq, action_seq = [], []
actions = ['standing_side_lunge', 'other']
cap = cv2.VideoCapture(
    'C:/Users/UCL7/VS_kwix/Byungsun_main/create_model_LSTM/side_crunch.mp4')

pose = set_cam()

while True:
    ret, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks is not None:
        landmark_subset = draw_function_KNN.custom_landmarks(results)

        joint, angle = get_points_angle(landmark_subset)

        d = np.concatenate([joint.flatten(), angle])
        seq.append(d)

        if len(seq) < seq_length:
            continue

        seq = np.array(seq)
        seq = seq.reshape(1, -1)

        y_pred = classifier.predict(seq)
        i_pred = int(np.argmax(y_pred))
        action = actions[i_pred]
        action_seq.append(action)

        # action, y_pred = input_data(seq, seq_length,classifier, actions)
        # action_seq.append(action)

        if len(action_seq) < 3:
            continue

        if action_seq[-1] == action_seq[-2]:
            if y_pred[0] > 0.8:
                rate = 0
                img = draw_function_KNN.drawing_Side_crunch(
                    img, landmark_subset.landmark, rate)
            elif y_pred[0] > 0.6:
                rate = 1
                img = draw_function_KNN.drawing_Side_crunch(
                    img, landmark_subset.landmark, rate)
            else:
                rate = 2
                img = draw_function_KNN.drawing_Side_crunch(
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

result = classifier.predict_proba(test_xdata)
print(result)