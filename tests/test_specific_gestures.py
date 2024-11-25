# tests/test_specific_gestures.py
import pytest
import numpy as np
import logging
from gesture_recognition import HandGestureRecognizer

class TestSpecificGestures:
    @pytest.fixture
    def recognizer(self):
        return HandGestureRecognizer()
    
    def test_ok_gesture(self, recognizer):
        landmarks = [[0, 0] for _ in range(21)]
        
        landmarks[4] = [0.3, 0.3]
        landmarks[8] = [0.3, 0.3]
        landmarks[12] = [0.5, 0.2]
        landmarks[16] = [0.6, 0.2]
        landmarks[20] = [0.7, 0.2]
        landmarks[0] = [0.5, 0.5]
        
        finger_states = [1, 1, 0, 0, 0]
        gesture, color = recognizer.recognize_gesture(finger_states, landmarks)
        
        assert gesture == "OK"
        assert color == recognizer.gestures["OK"]
    
    def test_u_gesture(self, recognizer):
        landmarks = [[0, 0] for _ in range(21)]
        
        landmarks[4] = [0.2, 0.3]
        landmarks[8] = [0.4, 0.2]
        landmarks[12] = [0.5, 0.2]
        landmarks[16] = [0.6, 0.5]
        landmarks[20] = [0.7, 0.5]
        
        finger_states = [1, 1, 1, 0, 0]
        gesture, color = recognizer.recognize_gesture(finger_states, landmarks)
        
        assert gesture == "U"
        assert color == recognizer.gestures["U"]
    
    def test_ok_gesture_with_variations(self, recognizer):
        test_positions = [
            ([0.3, 0.3], [0.31, 0.31]),
            ([0.3, 0.3], [0.29, 0.29]),
            ([0.25, 0.25], [0.25, 0.25])
        ]
        
        for thumb_pos, index_pos in test_positions:
            landmarks = [[0, 0] for _ in range(21)]
            landmarks[4] = thumb_pos
            landmarks[8] = index_pos
            
            finger_states = [1, 1, 0, 0, 0]
            gesture, _ = recognizer.recognize_gesture(finger_states, landmarks)
            assert gesture == "OK"
    
    def test_gesture_transition(self, recognizer):
        landmarks = [[0, 0] for _ in range(21)]
        transitions = [
            ([1, 1, 0, 0, 0], "OK"),
            ([1, 1, 1, 0, 0], "U"),
            ([0, 1, 1, 0, 0], "PEACE")
        ]
        
        for finger_states, expected_gesture in transitions:
            gesture, _ = recognizer.recognize_gesture(finger_states, landmarks)
            assert gesture == expected_gesture
    
    def test_invalid_gestures(self, recognizer):
        invalid_states = [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1]
        ]
        
        landmarks = [[0, 0] for _ in range(21)]
        
        for finger_states in invalid_states:
            gesture, _ = recognizer.recognize_gesture(finger_states, landmarks)
            assert gesture == "No gesture"