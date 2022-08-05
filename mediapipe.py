import mediapipe as mp
import cv2


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

image = cv2.imread('C:/Users/UCL7\Desktop/VS_kwix/409-2-1-19-Z21_C-0000011.jpg')


img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(img_rgb)
img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


while True:
    cv2.imshow('image', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break