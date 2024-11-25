# test_hand_gesture_recognizer.py
import pytest
import numpy as np
import cv2
import logging
import mediapipe as mp
from gesture_recognition import HandGestureRecognizer

class TestHandGestureRecognizer:
    @pytest.fixture
    def recognizer(self):
        return HandGestureRecognizer()

    def test_initialization(self, recognizer):
        logging.info("Testing HandGestureRecognizer initialization")
        assert isinstance(recognizer.hands, mp.solutions.hands.Hands)
        assert isinstance(recognizer.pose, mp.solutions.pose.Pose)
        assert isinstance(recognizer.face_mesh, mp.solutions.face_mesh.FaceMesh)
        assert len(recognizer.gestures) == 7

    def test_get_finger_state(self, recognizer):
        logging.info("Testing finger state detection")
        landmarks = [
            [0.5, 0.5], [0.4, 0.4], [0.3, 0.3], [0.2, 0.2], [0.1, 0.1],
            [0.5, 0.4], [0.5, 0.3], [0.5, 0.2], [0.5, 0.1], [0.6, 0.4],
            [0.6, 0.3], [0.6, 0.2], [0.6, 0.1], [0.7, 0.5], [0.7, 0.6],
            [0.7, 0.7], [0.7, 0.8], [0.8, 0.5], [0.8, 0.6], [0.8, 0.7],
            [0.8, 0.8]
        ]
        finger_states = recognizer.get_finger_state(landmarks)
        assert len(finger_states) == 5
        assert isinstance(finger_states, list)

    def test_calculate_distance(self, recognizer):
        logging.info("Testing distance calculation")
        point1 = (0, 0)
        point2 = (3, 4)
        distance = recognizer.calculate_distance(point1, point2)
        assert distance == 5.0

    def test_recognize_gesture_peace(self, recognizer):
        logging.info("Testing peace gesture recognition")
        finger_states = [0, 1, 1, 0, 0]
        landmarks = [[0, 0] for _ in range(21)]
        gesture, color = recognizer.recognize_gesture(finger_states, landmarks)
        assert gesture == "PEACE"
        assert color == recognizer.gestures["PEACE"]

    @pytest.mark.parametrize("frame_shape", [(480, 640, 3)])
    def test_process_frame(self, recognizer, frame_shape):
        logging.info(f"Testing frame processing with shape {frame_shape}")
        frame = np.zeros(frame_shape, dtype=np.uint8)
        processed_frame = recognizer.process_frame(frame)
        assert processed_frame.shape == frame_shape
        assert isinstance(processed_frame, np.ndarray)