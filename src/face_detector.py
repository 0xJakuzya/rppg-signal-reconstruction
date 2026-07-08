import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from src import config

class FaceDetector:
    def __init__(self):
        base_options = mp_python.BaseOptions(model_asset_path=str(config.face_model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

    def landmarks(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(image)
        if not result.face_landmarks:
            return None
        height, width = frame.shape[:2]
        points = []
        for landmark in result.face_landmarks[0]:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
        return points

    def face_crop(self, frame, landmarks, size, margin=config.face_margin_frac):
        points = np.array(landmarks, np.int32)
        height, width = frame.shape[:2]
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        x_max, y_max = points[:, 0].max(), points[:, 1].max()
        pad_x = int((x_max - x_min) * margin)
        pad_y = int((y_max - y_min) * margin)
        x0 = max(x_min - pad_x, 0)
        y0 = max(y_min - pad_y, 0)
        x1 = min(x_max + pad_x, width)
        y1 = min(y_max + pad_y, height)
        cropped = frame[y0:y1, x0:x1]
        if cropped.size == 0:
            return np.zeros((size, size, 3), np.uint8)
        return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)

    def close(self):
        self.landmarker.close()
