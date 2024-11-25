# tests/test_robustness.py
import pytest
import numpy as np
import cv2
import logging
import time
from gesture_recognition import HandGestureRecognizer

class TestRobustness:
    @pytest.fixture
    def recognizer(self):
        return HandGestureRecognizer()
    
    @pytest.fixture
    def corrupted_frame(self):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frame[240:241, :] = 0
        frame[:, 320:321] = 0
        return frame
    
    def test_camera_failure(self, recognizer, mock_video_capture):
        mock_video_capture.read.return_value = (False, None)
        mock_video_capture.isOpened.return_value = False
        
        with pytest.raises(Exception):
            recognizer.start_recognition()
        
        mock_video_capture.release.assert_called_once()
    
    def test_corrupted_frame_handling(self, recognizer):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 100:200] = np.nan
        
        processed_frame = recognizer.process_frame(frame)
        assert processed_frame is not None
        assert processed_frame.shape == frame.shape
    
    @pytest.mark.parametrize("resolution", [
        (320, 240, 3),
        (640, 480, 3),
        (1280, 720, 3),
        (1920, 1080, 3)
    ])
    def test_different_resolutions(self, recognizer, resolution):
        frame = np.zeros(resolution, dtype=np.uint8)
        processed_frame = recognizer.process_frame(frame)
        assert processed_frame.shape == resolution
    
    def test_memory_stability(self, recognizer):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frames = []
        
        for _ in range(10):
            processed = recognizer.process_frame(frame)
            frames.append(processed.copy())
        
        del frames
        
        final_frame = recognizer.process_frame(frame)
        assert final_frame is not None
    
    def test_processing_speed(self, recognizer):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processing_times = []
        
        for _ in range(10):
            start_time = time.time()
            recognizer.process_frame(frame)
            processing_times.append(time.time() - start_time)
        
        average_time = sum(processing_times) / len(processing_times)
        assert average_time < 0.1
    
    def test_real_time_performance(self, recognizer):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_count = 0
        start_time = time.time()
        
        while time.time() - start_time < 1.0:
            recognizer.process_frame(frame)
            frame_count += 1
        
        fps = frame_count / (time.time() - start_time)
        assert fps >= 15