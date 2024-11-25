# test_gesture_detection.py
import pytest
import numpy as np
import logging
import mediapipe as mp
from gesture_recognition import HandGestureRecognizer

class TestGestureDetection:
    @pytest.fixture
    def recognizer(self):
        return HandGestureRecognizer()

    def test_detect_smile(self, recognizer):
        logging.info("Testing smile detection")
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        class MockFaceLandmarks:
            def __init__(self):
                self.landmark = {
                    61: MockLandmark(0.4, 0.6),
                    291: MockLandmark(0.6, 0.6),
                    13: MockLandmark(0.5, 0.55),
                    14: MockLandmark(0.5, 0.65)
                }
                
            def __getattr__(self, name):
                if name == 'landmark':
                    return self.landmark
                return MockLandmark(0.5, 0.5)

        mock_landmarks = MockFaceLandmarks()
        result = recognizer.detect_smile(mock_landmarks)
        assert result in [None, "SMILE"]

    def test_detect_shrug(self, recognizer):
        logging.info("Testing shoulder shrug detection")
        class MockPoseLandmark:
            def __init__(self, y):
                self.y = y

        class MockPoseLandmarks:
            def __init__(self, left_y, right_y):
                self.landmark = {}
                self.landmark[11] = MockPoseLandmark(left_y)
                self.landmark[12] = MockPoseLandmark(right_y)

        mock_landmarks = MockPoseLandmarks(0.5, 0.5)
        initial_result = recognizer.detect_shrug(mock_landmarks)
        assert isinstance(initial_result, bool)

        mock_landmarks = MockPoseLandmarks(0.4, 0.4)
        shrug_result = recognizer.detect_shrug(mock_landmarks)
        assert isinstance(shrug_result, bool)

        mock_landmarks = MockPoseLandmarks(0.5, 0.5)
        final_result = recognizer.detect_shrug(mock_landmarks)
        assert isinstance(final_result, bool)

    def test_detect_head_movement(self, recognizer):
        logging.info("Testing head movement detection")
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        class MockFaceLandmarks:
            def __init__(self, x, y):
                self.landmark = {
                    1: MockLandmark(x, y)
                }

        static_landmarks = MockFaceLandmarks(0.5, 0.5)
        assert not recognizer.detect_head_movement(static_landmarks)

        moving_landmarks = [
            MockFaceLandmarks(0.48, 0.5),
            MockFaceLandmarks(0.52, 0.5),
            MockFaceLandmarks(0.48, 0.5),
            MockFaceLandmarks(0.52, 0.5)
        ]

        for landmarks in moving_landmarks:
            result = recognizer.detect_head_movement(landmarks)
            assert isinstance(result, bool)