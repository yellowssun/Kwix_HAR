import cv2, os
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils    
    
    
POSE_CONNECTIONS = frozenset([(1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
                                  (1, 7), (2, 8), (7, 8), (7, 9), (8, 10),
                                  (9, 11), (10, 12), (11, 13), (13, 15),
                                  (12, 14), (14, 16), (11, 15), (12, 16)])

def detect_mediapipe(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results
    
path_dir = 'D:/fitness_image_data/Training/crunch/1/C'
folder_list = os.listdir(path_dir)
print(folder_list)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        for i in folder_list:
            img = cv2.imread(path_dir+'/'+i)

            image, results = detect_mediapipe(img, pose)
            if results.pose_landmarks is not None:
                landmark_subset = landmark_pb2.NormalizedLandmarkList(
                                            landmark = [
                                                        results.pose_landmarks.landmark[0],
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

                mp_drawing.draw_landmarks(image, landmark_subset, POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1,circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,circle_radius=1)
                                    )

            cv2.imshow('Mediapipe Feed', image)
            # cv2.imwrite('1', image)

            if cv2.waitKey(10) & 0xFF == ord('q'): # break video
                break
