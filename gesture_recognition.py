import cv2
import mediapipe as mp
import numpy as np
import math

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.hand_drawing_spec = self.mp_draw.DrawingSpec(
            color=(0, 191, 255),
            thickness=2,
            circle_radius=2
        )
        
        self.connection_drawing_spec = self.mp_draw.DrawingSpec(
            color=(0, 191, 255),
            thickness=2
        )
        
        self.gestures = {
            "PEACE": (0, 255, 128)
        }

    def get_finger_state(self, landmarks):
        fingers = []
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]
        thumb_raised = thumb_tip[0] < thumb_base[0]
        fingers.append(1 if thumb_raised else 0)
        
        for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
            finger_raised = landmarks[tip][1] < landmarks[pip][1]
            fingers.append(1 if finger_raised else 0)
        
        return fingers
    
    def recognize_gesture(self, finger_states, landmarks):
        index_up = finger_states[1]
        middle_up = finger_states[2]
        
        index_tip = landmarks[8][1]
        index_base = landmarks[5][1]
        middle_tip = landmarks[12][1]
        middle_base = landmarks[9][1]
        
        if index_up and middle_up:
            index_raised = (index_base - index_tip) > 0.1
            middle_raised = (middle_base - middle_tip) > 0.1
            
            if index_raised and middle_raised:
                return "PEACE", self.gestures["PEACE"]
        
        return "No gesture", (255, 255, 255)

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_results = self.hands.process(rgb_frame)
        
        detected_gesture = None
        
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.hand_drawing_spec,
                    connection_drawing_spec=self.connection_drawing_spec
                )
                
                landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                finger_states = self.get_finger_state(landmarks)
                hand_gesture, _ = self.recognize_gesture(finger_states, landmarks)
                if hand_gesture != "No gesture":
                    detected_gesture = hand_gesture
        
        if detected_gesture:
            frame = self.create_gesture_overlay(frame, detected_gesture, self.gestures[detected_gesture])
        
        return frame

    def create_gesture_overlay(self, frame, gesture, color):
        overlay = frame.copy()
        padding = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        text = f"{gesture}"
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        rect_width = text_width + (padding * 2)
        rect_height = text_height + (padding * 2)
        
        cv2.rectangle(overlay, 
                    (10, 10), 
                    (10 + rect_width, 10 + rect_height), 
                    (0, 0, 0), 
                    -1)
        
        cv2.rectangle(overlay, 
                    (10, 10), 
                    (10 + rect_width, 10 + rect_height), 
                    color, 
                    3)
        
        cv2.putText(
            overlay,
            text,
            (10 + padding, 10 + text_height + (padding // 2)),
            font,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA
        )
        
        alpha = 0.9
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
        
    def start_recognition(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Gesture Recognition', cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)
            
            cv2.imshow('Gesture Recognition', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = HandGestureRecognizer()
    recognizer.start_recognition()