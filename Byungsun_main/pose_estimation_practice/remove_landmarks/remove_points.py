import cv2, os, time
import numpy as np
import matplotlib.pyplot as plt
import redefine_pose_connection, redefine_pose
import redefine_holistic, redefine_solution_base
import redefine_drawing_utils, redefine_drawing_styles
from mediapipe.framework.formats import landmark_pb2

mp_holistic = redefine_holistic
mp_drawing = redefine_drawing_utils


def detect_mediapipe(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        image, results = detect_mediapipe(frame, holistic) 
        
        """landmark를 다시 지정하는 코드로 
            landmark가 캠에 존재하지 않으면 에러 생김
            임시적으로 만든 코드(landmark가 캠에 존재하지 않으면 detecting을 멈춘다)"""
        try:
            landmark_subset = landmark_pb2.NormalizedLandmarkList(
                                    landmark = [results.pose_landmarks.landmark[0],
                                                results.pose_landmarks.landmark[11],
                                                results.pose_landmarks.landmark[12],
                                                results.pose_landmarks.landmark[13],
                                                results.pose_landmarks.landmark[14],
                                                results.pose_landmarks.landmark[15],
                                                results.pose_landmarks.landmark[16],
                                                results.pose_landmarks.landmark[23],
                                                results.pose_landmarks.landmark[24],
                                                results.pose_landmarks.landmark[25],
                                                results.pose_landmarks.landmark[26],
                                                results.pose_landmarks.landmark[27],
                                                results.pose_landmarks.landmark[28],
                                                results.pose_landmarks.landmark[29],
                                                results.pose_landmarks.landmark[30],
                                                results.pose_landmarks.landmark[31],
                                                results.pose_landmarks.landmark[32]]
                                    )
        except:
            pass
        
        mp_drawing.draw_landmarks(image, landmark_subset, redefine_pose_connection.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=2,circle_radius=4),
                        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=2,circle_radius=4)
                        )     

        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1,circle_radius=1),
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1,circle_radius=1)
                        )
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1,circle_radius=1),
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1,circle_radius=1)
                        )

        print(results.pose_landmarks)
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'): # break video
            break

cap.release()
cv2.destroyAllWindows()