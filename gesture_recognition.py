import cv2
import mediapipe as mp
import numpy as np
import math
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
            "NO SE": (255, 0, 0),
            "SONRISA": (255, 255, 0),
            "PENSANDO": (191, 255, 0)
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
            return "SONRISA"
            
        return None
    
    def detect_reflection_gesture(self, face_landmarks, hand_landmarks):
        if not face_landmarks or not hand_landmarks:
            return False
            
        chin = face_landmarks.landmark[152]
        
        left_eyebrow_points = [face_landmarks.landmark[i] for i in [70, 46, 53, 52, 65]]
        right_eyebrow_points = [face_landmarks.landmark[i] for i in [300, 276, 283, 282, 295]]
        
        left_eye_top = face_landmarks.landmark[159].y
        right_eye_top = face_landmarks.landmark[386].y
        neutral_position = (left_eye_top + right_eye_top) / 2
        
        left_eyebrow_y = sum(point.y for point in left_eyebrow_points) / len(left_eyebrow_points)
        right_eyebrow_y = sum(point.y for point in right_eyebrow_points) / len(right_eyebrow_points)
        current_eyebrow_y = (left_eyebrow_y + right_eyebrow_y) / 2
        
        eyebrow_displacement = neutral_position - current_eyebrow_y
        
        self.eyebrow_positions.append(current_eyebrow_y)
        
        hand_chin_distances = [
            self.calculate_distance(
                (hand_landmark.x, hand_landmark.y),
                (chin.x, chin.y)
            )
            for hand_landmark in hand_landmarks.landmark
        ]
        min_hand_chin_distance = min(hand_chin_distances)
        
        if len(self.eyebrow_positions) >= 2:
            eyebrow_movement = self.eyebrow_positions[-1] - self.eyebrow_positions[0]
            
            eyebrows_raised = (
                eyebrow_movement < -self.eyebrow_raise_threshold * 0.5 or
                eyebrow_displacement > self.eyebrow_raise_threshold * 2
            )
            
            hand_near_chin = min_hand_chin_distance < self.chin_distance_threshold * 1.8
            
            if eyebrows_raised and hand_near_chin:
                return True
            elif not hand_near_chin:
                return False
            else:
                return (
                    eyebrow_displacement > 0 or
                    abs(eyebrow_movement) > self.eyebrow_raise_threshold * 0.3
                )
        
        return False
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        pose_results = self.pose.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        detected_gesture = None
        
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            
            key_face_points = [
                61, 291, 13, 14, 33, 133, 362, 263,
                152,
                107, 336,
                71, 301
            ]
            
            for idx in key_face_points:
                point = face_landmarks.landmark[idx]
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            smile_gesture = self.detect_smile(face_landmarks)
            if smile_gesture:
                detected_gesture = smile_gesture
            
            if hand_results.multi_hand_landmarks:
                if self.detect_reflection_gesture(face_landmarks, hand_results.multi_hand_landmarks[0]):
                    detected_gesture = "PENSANDO"
        
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
                detected_gesture = "NO SE"
        
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
    
    def recognize_gesture(self, finger_states, landmarks):
        thumb_up = finger_states[0]
        index_up = finger_states[1]
        middle_up = finger_states[2]
        ring_up = finger_states[3]
        pinky_up = finger_states[4]
        
        index_tip = landmarks[8][1]
        index_base = landmarks[5][1]
        middle_tip = landmarks[12][1]
        middle_base = landmarks[9][1]
        
        if index_up and middle_up:
            index_raised = (index_base - index_tip) > 0.1
            middle_raised = (middle_base - middle_tip) > 0.1
            
            if index_raised and middle_raised:
                return "PEACE", self.gestures["PEACE"]
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = self.calculate_distance(thumb_tip, index_tip)
        
        if distance < 0.1:
            return "OK", self.gestures["OK"]
        
        return "No gesture", (255, 255, 255)

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
            info_text = "Gestos disponibles:"
            cv2.putText(processed_frame, info_text, 
                    (10, height - 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, (255, 255, 255), 2)
            
            gestures_info = [
                "THINKING: Cejas levantadas + Mano en mentón",
                "PEACE: Dedos índice y medio levantados",
                "OK: Pulgar e índice formando círculo",
                "NO SE: Encogimiento de hombros",
                "SONRISA: Sonrisa amplia detectada"
            ]
            
            for i, text in enumerate(gestures_info):
                cv2.putText(processed_frame, text, 
                           (20, height - 90 + (i * 20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (200, 200, 200), 1)
            
            if len(self.eyebrow_positions) >= 2:
                eyebrow_movement = self.eyebrow_positions[-1] - self.eyebrow_positions[0]
                if eyebrow_movement < -self.eyebrow_raise_threshold:
                    cv2.putText(processed_frame, "Cejas Levantadas", 
                              (width - 200, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.6, (0, 255, 0), 2)
            
            cv2.imshow('Gesture Recognition', processed_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.eyebrow_positions.clear()
                
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = HandGestureRecognizer()
    recognizer.start_recognition()