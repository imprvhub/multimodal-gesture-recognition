import cv2
import mediapipe as mp
import numpy as np
import math
import time
from collections import deque

class HandGestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            refine_landmarks=True
        )
        
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_drawing_spec = self.mp_draw.DrawingSpec(
            thickness=1,
            circle_radius=1,
            color=(0, 255, 0)
        )
        
        self.face_connection_spec = self.mp_draw.DrawingSpec(
            color=(0, 255, 0),
            thickness=1
        )
        
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
            "PEACE": (0, 255, 128),
            "OK": (255, 128, 0),
            "U": (0, 255, 255),
            "IDK": (255, 0, 0),
            "SMILE": (255, 255, 0),
            "THINKING": (191, 255, 0),
            "GROOVE": (147, 20, 255)
        }
        
        self.shoulder_heights = deque(maxlen=10)
        self.neutral_shoulder_height = None
        self.shrug_threshold = 0.03
        self.shrug_counter = 0
        self.shrug_frames_required = 3
        self.last_shrug_state = False
        
        self.smile_threshold = 0.25
        self.mouth_height_threshold = 0.02
        
        self.chin_distance_threshold = 0.15
        self.eyebrow_raise_threshold = 0.01
        
        self.eyebrow_positions = deque(maxlen=5)
        
        self.head_positions = deque(maxlen=5)
        self.last_direction = None
        self.direction_changes = 0
        self.last_detection_time = 0
        self.groove_cooldown = 0
            
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
    
    def calculate_distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_smile(self, face_landmarks):
        if not face_landmarks:
            return None
            
        mouth_left = face_landmarks.landmark[61]
        mouth_right = face_landmarks.landmark[291]
        mouth_top = face_landmarks.landmark[13]
        mouth_bottom = face_landmarks.landmark[14]
        
        mouth_width = self.calculate_distance(
            (mouth_left.x, mouth_left.y),
            (mouth_right.x, mouth_right.y)
        )
        
        mouth_height = self.calculate_distance(
            (mouth_top.x, mouth_top.y),
            (mouth_bottom.x, mouth_bottom.y)
        )
        
        smile_ratio = mouth_width / (mouth_height + 1e-6)
        
        if smile_ratio > self.smile_threshold and mouth_height > self.mouth_height_threshold:
            return "SMILE"
            
        return None
    
    def detect_head_movement(self, face_landmarks):
        if not face_landmarks:
            return False
            
        nose_tip = face_landmarks.landmark[1]
        current_pos = (nose_tip.x, nose_tip.y)
        
        if not hasattr(self, 'head_positions'):
            self.head_positions = deque(maxlen=5)
            self.last_direction = None
            self.direction_changes = 0
            self.last_detection_time = 0
        
        self.head_positions.append(current_pos)
        
        if len(self.head_positions) < 3:
            return False
        
        current_direction = None
        total_movement = 0
        
        for i in range(len(self.head_positions) - 1):
            dx = self.head_positions[i+1][0] - self.head_positions[i][0]
            dy = self.head_positions[i+1][1] - self.head_positions[i][1]
            
            movement = abs(dx) * 1.5 + abs(dy) * 0.5
            total_movement += movement
            
            if abs(dx) > 0.005:
                current_direction = 1 if dx > 0 else -1
        
        if (self.last_direction is not None and 
            current_direction is not None and 
            current_direction != self.last_direction):
            self.direction_changes += 1
            
        self.last_direction = current_direction
        
        current_time = cv2.getTickCount() / cv2.getTickFrequency()
        time_since_last = current_time - self.last_detection_time
        
        if (total_movement > 0.02 and
            self.direction_changes >= 1 and
            time_since_last > 0.5):
            
            self.last_detection_time = current_time
            self.direction_changes = 0
            return True
            
        if total_movement < 0.01:
            self.direction_changes = 0
            
        return False

    def detect_reflection_gesture(self, face_landmarks, hand_landmarks):
        if not face_landmarks or not hand_landmarks:
            return False
                
        chin = face_landmarks.landmark[152]

        hand_chin_distances = [
            self.calculate_distance(
                (hand_landmark.x, hand_landmark.y),
                (chin.x, chin.y)
            )
            for hand_landmark in hand_landmarks.landmark
        ]
        min_hand_chin_distance = min(hand_chin_distances)

        self.hand_in_thinking_position = min_hand_chin_distance < self.chin_distance_threshold
        return self.hand_in_thinking_position
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)

        detected_gestures = []

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]

            key_face_points = [
                61, 291, 13, 14, 33, 133, 362, 263,
                152,
                107, 336,
                71, 301,
                4
            ]
            
            for idx in key_face_points:
                point = face_landmarks.landmark[idx]
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if self.detect_head_movement(face_landmarks):
                self.groove_cooldown = 5
                detected_gestures.append(("GROOVE", self.gestures["GROOVE"]))
            elif self.groove_cooldown > 0:
                self.groove_cooldown -= 1
                if self.groove_cooldown > 0:
                    detected_gestures.append(("GROOVE", self.gestures["GROOVE"]))

            smile_gesture = self.detect_smile(face_landmarks)
            if smile_gesture:
                detected_gestures.append((smile_gesture, self.gestures[smile_gesture]))
            
            if hand_results.multi_hand_landmarks:
                if self.detect_reflection_gesture(face_landmarks, hand_results.multi_hand_landmarks[0]):
                    detected_gestures.append(("THINKING", self.gestures["THINKING"]))

        if pose_results.pose_landmarks:
            shoulders = [
                pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER],
                pose_results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            ]
            
            for shoulder in shoulders:
                x = int(shoulder.x * frame.shape[1])
                y = int(shoulder.y * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            if self.detect_shrug(pose_results.pose_landmarks):
                detected_gestures.append(("IDK", self.gestures["IDK"]))

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.hand_drawing_spec,
                    self.connection_drawing_spec
                )
                
                landmarks = [[lm.x, lm.y] for lm in hand_landmarks.landmark]
                finger_states = self.get_finger_state(landmarks)
                hand_gesture, color = self.recognize_gesture(finger_states, landmarks)
                if hand_gesture != "No gesture":
                    detected_gestures.append((hand_gesture, color))

        if detected_gestures:
            detected_gestures.sort(key=lambda x: x[0])
            for i, (gesture, color) in enumerate(detected_gestures[:2]):
                frame = self.create_gesture_overlay(frame, gesture, color, i)

        return frame
    
    def recognize_gesture(self, finger_states, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        
        if hasattr(self, 'hand_in_thinking_position') and self.hand_in_thinking_position:
            return "No gesture", (255, 255, 255)
        
        if distance < 0.1:
            middle_tip = landmarks[12][1]
            ring_tip = landmarks[16][1]
            pinky_tip = landmarks[20][1]
            wrist = landmarks[0][1]
            
            if (middle_tip < wrist and ring_tip < wrist and pinky_tip < wrist):
                return "OK", self.gestures["OK"]
        
        if finger_states[1] and finger_states[2] and not finger_states[3] and not finger_states[4]:
            return "PEACE", self.gestures["PEACE"]
            
        return "No gesture", (255, 255, 255)

    def create_gesture_overlay(self, frame, gesture, color, position=0):
        overlay = frame.copy()
        padding = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_thickness = 3
        text = f"{gesture}"
        
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        rect_width = text_width + (padding * 2)
        rect_height = text_height + (padding * 2)

        y_offset = position * (rect_height + 10)
        
        cv2.rectangle(overlay, 
                    (10, 10 + y_offset), 
                    (10 + rect_width, 10 + rect_height + y_offset), 
                    (0, 0, 0), 
                    -1)
        
        cv2.rectangle(overlay, 
                    (10, 10 + y_offset), 
                    (10 + rect_width, 10 + rect_height + y_offset), 
                    color, 
                    3)
        
        cv2.putText(
            overlay,
            text,
            (10 + padding, 10 + text_height + (padding // 2) + y_offset),
            font,
            font_scale,
            color,
            font_thickness,
            cv2.LINE_AA
        )
        
        alpha = 0.9
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame
    
    def detect_shrug(self, pose_landmarks):
        if not pose_landmarks:
            return False
            
        left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        current_height = (left_shoulder + right_shoulder) / 2
        
        self.shoulder_heights.append(current_height)
        
        if self.neutral_shoulder_height is None and len(self.shoulder_heights) >= 5:
            self.neutral_shoulder_height = sum(list(self.shoulder_heights)[:-1]) / (len(self.shoulder_heights) - 1)
        
        if self.neutral_shoulder_height is None:
            return False
        
        height_diff = self.neutral_shoulder_height - current_height
        shoulders_raised = height_diff > self.shrug_threshold
        
        if shoulders_raised:
            self.shrug_counter = min(self.shrug_counter + 1, self.shrug_frames_required)
        else:
            self.shrug_counter = max(self.shrug_counter - 1, 0)
            self.neutral_shoulder_height = self.neutral_shoulder_height * 0.95 + current_height * 0.05
        
        shrug_detected = self.shrug_counter >= self.shrug_frames_required
        
        if shrug_detected and not self.last_shrug_state:
            self.neutral_shoulder_height = None
        
        self.last_shrug_state = shrug_detected
        
        return shrug_detected
        
    def start_recognition(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow('Gesture Recognition', cv2.WINDOW_NORMAL)
        
        face_mesh_drawing_spec = self.mp_draw.DrawingSpec(
            thickness=1,
            circle_radius=1,
            color=(0, 255, 0)
        )
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            processed_frame = self.process_frame(frame)
            
            height, width = frame.shape[:2]
            info_text = "Available Gestures:"
            cv2.putText(processed_frame, info_text, 
                    (10, height - 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
            
            gestures_info = [
                "THINKING: Raised eyebrows + Hand on chin",
                "PEACE: Index and middle fingers raised",
                "OK: Thumb and index forming circle",
                "IDK: Shoulder shrug",
                "SMILE: Wide smile detected",
                "GROOVE: Move head to the rhythm"
            ]
            
            for i, text in enumerate(gestures_info):
                cv2.putText(processed_frame, text, 
                           (20, height - 90 + (i * 20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (200, 200, 200), 1)
            
            if len(self.eyebrow_positions) >= 2:
                eyebrow_movement = self.eyebrow_positions[-1] - self.eyebrow_positions[0]
                if eyebrow_movement < -self.eyebrow_raise_threshold:
                    cv2.putText(processed_frame, "Eyebrows Raised",
                            (width - 200, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)
            
            cv2.imshow('Gesture Recognition', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.eyebrow_positions.clear()
                self.head_positions.clear()
                self.groove_counter = 0
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = HandGestureRecognizer()
    recognizer.start_recognition()
