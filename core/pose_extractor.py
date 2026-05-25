"""MediaPipe Pose Extraction Module
Extracts 33-point skeleton sequences from video/webcam
"""

import cv2
import mediapipe as mp
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple

class PoseExtractor:
    """Extract pose landmarks from video using MediaPipe"""
    
    def __init__(self, confidence_threshold=0.5):
        """Initialize MediaPipe Pose detector"""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=confidence_threshold,
            min_tracking_confidence=confidence_threshold
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.landmark_names = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow',
            'right_elbow', 'left_wrist', 'right_wrist',
            'left_pinky', 'right_pinky', 'left_index', 'right_index',
            'left_thumb', 'right_thumb', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
            'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index'
        ]
    
    def extract_sequence(self, video_path: str) -> List[Dict]:
        """Extract pose sequence from video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = self._landmarks_to_dict(results.pose_landmarks)
                frames.append({
                    'frame_id': frame_count,
                    'landmarks': landmarks,
                    'timestamp': frame_count / cap.get(cv2.CAP_PROP_FPS)
                })
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def _landmarks_to_dict(self, pose_landmarks) -> Dict:
        """Convert MediaPipe landmarks to dictionary"""
        landmarks_dict = {}
        for idx, landmark in enumerate(pose_landmarks.landmark):
            landmarks_dict[self.landmark_names[idx]] = {
                'x': float(landmark.x),
                'y': float(landmark.y),
                'z': float(landmark.z),
                'visibility': float(landmark.visibility)
            }
        return landmarks_dict
    
    def save_sequence(self, frames: List[Dict], output_path: str):
        """Save extracted sequence to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(frames, f, indent=2)
        print(f"✅ Saved {len(frames)} frames to {output_path}")
    
    def load_sequence(self, json_path: str) -> List[Dict]:
        """Load sequence from JSON file"""
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def get_body_region(self, landmarks: Dict, region: str) -> Dict:
        """Extract specific body region (arms, torso, legs)"""
        regions = {
            'arms': ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
            'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
            'legs': ['left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
            'hands': ['left_wrist', 'right_wrist', 'left_pinky', 'right_pinky', 'left_index', 'right_index']
        }
        
        if region not in regions:
            raise ValueError(f"Unknown region: {region}. Available: {list(regions.keys())}")
        
        return {name: landmarks[name] for name in regions[region] if name in landmarks}