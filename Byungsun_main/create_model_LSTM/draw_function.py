import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils

"""랜드마크를 지우고 운동 종목에 따라 실제로 웹에 보이는 connection들 함수
    모델의 확률에 따라 connection의 색이 초록-빨강-검정으로 바뀜
    """


def custom_landmarks_17(results):
    """
    17개의 Landmarks
    """
    landmark_subset = landmark_pb2.NormalizedLandmarkList(
        landmark=[results.pose_landmarks.landmark[0]] +
        results.pose_landmarks.landmark[11: 17] +
        results.pose_landmarks.landmark[23:]
    )
    return landmark_subset


def set_color(rate):
    if rate == 0:
        color = (0, 255, 0)
    elif rate == 1:
        color = (0, 0, 255)
    else:
        color = (0, 0, 0)

    return color


def set_cam():
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return pose


def set_drawing(img, landmarks, pose_connection, color):
    mp_drawing.draw_landmarks(img, landmarks, pose_connection,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4))
    return img


POSE_CONNECTIONS_ALL = frozenset([(1, 2), (1, 3), (3, 5), (2, 4), (4, 6),
                                  (1, 7), (2, 8), (7, 8), (7, 9), (8, 10),
                                  (9, 11), (10, 12)])

POSE_CONNECTIONS_Crunch = frozenset(
    [(0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7)])

POSE_CONNECTIONS_Leg_raise = frozenset([(0, 2), (2, 3), (1, 3), (2, 4), (
    4, 6), (3, 5)])

POSE_CONNECTIONS_Side_lunge = frozenset(
    [(0, 1), (0, 2), (2, 4), (4, 6), (1, 3), (3, 5), (5, 7)])


POSE_CONNECTIONS_Knee_up = frozenset(
    [(0, 1), (0, 2), (2, 4), (4, 6), (1, 3), (3, 5)])

POSE_CONNECTIONS_Side_crunch = frozenset([(0, 2), (0, 4), (1, 3), (1, 5), (4, 6), (6, 8),
                                          (5, 7), (7, 9)])


def drawing_All(img, landmarks):
    color = (255, 255, 255)
    mp_drawing.draw_landmarks(img, landmarks, POSE_CONNECTIONS_ALL)
    img = set_drawing(img, landmarks, POSE_CONNECTIONS_ALL, color)
    return img


def drawing_Crunch(img, landmarks, rate):
    color = set_color(rate)

    landmarks1 = landmarks[1:3]
    landmarks2 = landmarks[7:13]
    _landmarks = landmarks1 + landmarks2
    landmarks = landmark_pb2.NormalizedLandmarkList(landmark=_landmarks)

    img = set_drawing(img, landmarks, POSE_CONNECTIONS_Crunch, color)
    return img


def drawing_Side_lunge(img, landmarks, rate):
    color = set_color(rate)

    _landmarks = landmarks[7:15]
    landmarks = landmark_pb2.NormalizedLandmarkList(landmark=_landmarks)
    img = set_drawing(img, landmarks, POSE_CONNECTIONS_Side_lunge, color)
    return img


def drawing_Leg_raise(img, landmarks, rate):
    color = set_color(rate)

    landmarks1 = landmarks[1:3]
    landmarks2 = landmarks[7:13]
    _landmarks = landmarks1 + landmarks2
    landmarks = landmark_pb2.NormalizedLandmarkList(landmark=_landmarks)
    img = set_drawing(img, landmarks, POSE_CONNECTIONS_Leg_raise, color)
    return img


def drawing_Knee_up(img, landmarks, rate):
    color = set_color(rate)

    _landmarks = landmarks[7:13]
    landmarks = landmark_pb2.NormalizedLandmarkList(landmark=_landmarks)
    img = set_drawing(img, landmarks, POSE_CONNECTIONS_Knee_up, color)
    return img


def drawing_Side_crunch(img, landmarks, rate):
    color = set_color(rate)

    landmarks1 = landmarks[1:5]
    landmarks2 = landmarks[7:13]
    _landmarks = landmarks1 + landmarks2
    landmarks = landmark_pb2.NormalizedLandmarkList(landmark=_landmarks)
    img = set_drawing(img, landmarks, POSE_CONNECTIONS_Side_crunch, color)
    return img
