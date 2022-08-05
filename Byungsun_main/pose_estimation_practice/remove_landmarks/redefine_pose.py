import enum
from typing import NamedTuple

import numpy as np


from mediapipe.calculators.core import constant_side_packet_calculator_pb2
from mediapipe.calculators.core import gate_calculator_pb2
from mediapipe.calculators.core import split_vector_calculator_pb2
from mediapipe.calculators.image import warp_affine_calculator_pb2
from mediapipe.calculators.tensor import image_to_tensor_calculator_pb2
from mediapipe.calculators.tensor import inference_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_classification_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_detections_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_landmarks_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_segmentation_calculator_pb2
from mediapipe.calculators.tflite import ssd_anchors_calculator_pb2
from mediapipe.calculators.util import detections_to_rects_calculator_pb2
from mediapipe.calculators.util import landmarks_smoothing_calculator_pb2
from mediapipe.calculators.util import local_file_contents_calculator_pb2
from mediapipe.calculators.util import logic_calculator_pb2
from mediapipe.calculators.util import non_max_suppression_calculator_pb2
from mediapipe.calculators.util import rect_transformation_calculator_pb2
from mediapipe.calculators.util import thresholding_calculator_pb2
from mediapipe.calculators.util import visibility_smoothing_calculator_pb2
from mediapipe.framework.tool import switch_container_pb2
# pylint: enable=unused-import
from redefine_solution_base import SolutionBase
from mediapipe.python.solutions import download_utils
# pylint: disable=unused-import
from redefine_pose_connection import POSE_CONNECTIONS
# pylint: enable=unused-import


class PoseLandmark(enum.IntEnum):
  """The 33 pose landmarks."""
  NOSE = 0
  ## LEFT_EYE_INNER = 1
  ## LEFT_EYE = 2
  ## LEFT_EYE_OUTER = 3
  ## RIGHT_EYE_INNER = 4
  ## RIGHT_EYE = 5
  ## RIGHT_EYE_OUTER = 6
  ## LEFT_EAR = 7
  ## RIGHT_EAR = 8
  ## MOUTH_LEFT = 9
  ## MOUTH_RIGHT = 10
  LEFT_SHOULDER = 11
  RIGHT_SHOULDER = 12
  LEFT_ELBOW = 13
  RIGHT_ELBOW = 14
  LEFT_WRIST = 15
  RIGHT_WRIST = 16
  # LEFT_PINKY = 17
  # RIGHT_PINKY = 18
  # LEFT_INDEX = 19
  # RIGHT_INDEX = 20
  # LEFT_THUMB = 21
  # RIGHT_THUMB = 22
  LEFT_HIP = 23
  RIGHT_HIP = 24
  LEFT_KNEE = 25
  RIGHT_KNEE = 26
  LEFT_ANKLE = 27
  RIGHT_ANKLE = 28
  LEFT_HEEL = 29
  RIGHT_HEEL = 30
  LEFT_FOOT_INDEX = 31
  RIGHT_FOOT_INDEX = 32


_BINARYPB_FILE_PATH = 'mediapipe/modules/pose_landmark/pose_landmark_cpu.binarypb'


def _download_oss_pose_landmark_model(model_complexity):
  """Downloads the pose landmark lite/heavy model from the MediaPipe Github repo if it doesn't exist in the package."""

  if model_complexity == 0:
    download_utils.download_oss_model(
        'mediapipe/modules/pose_landmark/pose_landmark_lite.tflite')
  elif model_complexity == 2:
    download_utils.download_oss_model(
        'mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite')


class Pose(SolutionBase):
  def __init__(self,
               static_image_mode=False,
               model_complexity=1,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):


    _download_oss_pose_landmark_model(model_complexity)
    super().__init__(
        binary_graph_path=_BINARYPB_FILE_PATH,
        side_inputs={
            'model_complexity': model_complexity,
            'smooth_landmarks': smooth_landmarks and not static_image_mode,
            'enable_segmentation': enable_segmentation,
            'smooth_segmentation':
                smooth_segmentation and not static_image_mode,
            'use_prev_landmarks': not static_image_mode,
        },
        calculator_params={
            'posedetectioncpu__TensorsToDetectionsCalculator.min_score_thresh':
                min_detection_confidence,
            'poselandmarkbyroicpu__tensorstoposelandmarksandsegmentation__ThresholdingCalculator.threshold':
                min_tracking_confidence,
        },
        outputs=['pose_landmarks', 'pose_world_landmarks', 'segmentation_mask'])

  def process(self, image: np.ndarray) -> NamedTuple:
    results = super().process(input_data={'image': image})
    if results.pose_landmarks:
      for landmark in results.pose_landmarks.landmark:
        landmark.ClearField('presence')
    if results.pose_world_landmarks:
      for landmark in results.pose_world_landmarks.landmark:
        landmark.ClearField('presence')
    return results