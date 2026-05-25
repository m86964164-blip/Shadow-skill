"""Real-Time Visualization with Ghost Overlay
Renders master pose as semi-transparent overlay on apprentice video
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import mediapipe as mp
from datetime import datetime
import json

class Visualizer:
    """Visualize pose comparison with ghost overlay"""
    
    def __init__(self, master_frames: List[Dict], similarity_results=None):
        """Initialize visualizer with master sequence"""
        self.master_frames = master_frames
        self.similarity_results = similarity_results or {}
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.session_data = []
    
    def render_ghost_overlay(self, user_frame, master_frame, comparison_result=None) -> np.ndarray:
        """Render master pose as semi-transparent ghost on user frame"""
        output = user_frame.copy()
        h, w = user_frame.shape[:2]
        
        # Draw master skeleton (ghost - 40% opacity)
        ghost_overlay = user_frame.copy()
        self._draw_skeleton(ghost_overlay, master_frame['landmarks'], color=(0, 255, 255), thickness=2, label='Master')
        cv2.addWeighted(ghost_overlay, 0.4, output, 0.6, 0, output)
        
        # Draw user skeleton (solid)
        self._draw_skeleton(output, user_frame['landmarks'], color=(0, 255, 0), thickness=3, label='You')
        
        # Draw HUD
        if comparison_result:
            self._draw_hud(output, comparison_result)
        
        return output
    
    def _draw_skeleton(self, frame, landmarks: Dict, color: Tuple, thickness: int, label: str = ''):
        """Draw pose skeleton on frame"""
        # Define skeleton connections
        connections = [
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle')
        ]
        
        h, w = frame.shape[:2]
        
        # Draw connections (bones)
        for start, end in connections:
            if start in landmarks and end in landmarks:
                start_pos = landmarks[start]
                end_pos = landmarks[end]
                
                x1, y1 = int(start_pos['x'] * w), int(start_pos['y'] * h)
                x2, y2 = int(end_pos['x'] * w), int(end_pos['y'] * h)
                
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw joints (circles)
        for joint_name, joint_data in landmarks.items():
            if joint_data.get('visibility', 0) > 0.5:
                x, y = int(joint_data['x'] * w), int(joint_data['y'] * h)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 4, color, -1)
    
    def _draw_hud(self, frame, comparison_result):
        """Draw heads-up display with score and feedback"""
        h, w = frame.shape[:2]
        
        # Overall score background
        cv2.rectangle(frame, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 100), (255, 255, 255), 2)
        
        # Score text
        score = comparison_result.overall_score
        if score > 85:
            color = (0, 255, 0)  # Green
        elif score > 70:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.putText(frame, f"Score: {score:.1f}%", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, comparison_result.feedback, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Per-joint heatmap (bottom right)
        joint_names = ['Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle']
        joint_keys = ['left_shoulder', 'left_elbow', 'left_wrist', 'left_hip', 'left_knee', 'left_ankle']
        
        y_offset = h - 150
        cv2.putText(frame, "Joint Quality:", (w - 200, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        for idx, (name, key) in enumerate(zip(joint_names, joint_keys)):
            score = comparison_result.joint_scores.get(key, 0)
            if score > 80:
                color = (0, 255, 0)
            elif score > 60:
                color = (0, 255, 255)
            else:
                color = (0, 0, 255)
            
            cv2.putText(frame, f"{name}: {score:.0f}%", (w - 200, y_offset + 25 + idx * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def render_real_time_session(self, user_video_path: str, master_video_path: str,
                                similarity_engine, output_path: str = None):
        """Process entire session with real-time overlay"""
        from core.pose_extractor import PoseExtractor
        
        extractor = PoseExtractor()
        master_frames = extractor.extract_sequence(master_video_path)
        
        cap = cv2.VideoCapture(user_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        session_scores = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp.solutions.pose.Pose().process(frame_rgb)
            
            if results.pose_landmarks and frame_count < len(master_frames):
                user_landmarks = self._landmarks_to_dict(results.pose_landmarks)
                user_frame = {'landmarks': user_landmarks, 'frame_id': frame_count}
                
                master_frame = master_frames[min(frame_count, len(master_frames) - 1)]
                
                # Compare
                comparison = similarity_engine.compare_sequences([master_frame], [user_frame])
                session_scores.append(comparison.overall_score)
                
                # Render
                output_frame = self.render_ghost_overlay(frame, master_frame, comparison)
                
                if output_path:
                    out.write(output_frame)
                
                cv2.imshow('Shadow Skill - Real-Time', output_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        return session_scores
    
    def _landmarks_to_dict(self, pose_landmarks) -> Dict:
        """Convert MediaPipe landmarks to dictionary"""
        landmark_names = [
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
        landmarks_dict = {}
        for idx, landmark in enumerate(pose_landmarks.landmark):
            landmarks_dict[landmark_names[idx]] = {
                'x': float(landmark.x),
                'y': float(landmark.y),
                'z': float(landmark.z),
                'visibility': float(landmark.visibility)
            }
        return landmarks_dict
    
    def generate_session_report(self, session_scores: List[float], joint_scores: Dict[str, float]) -> Dict:
        """Generate comprehensive session report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'session_stats': {
                'average_score': np.mean(session_scores) if session_scores else 0,
                'max_score': max(session_scores) if session_scores else 0,
                'min_score': min(session_scores) if session_scores else 0,
                'total_frames': len(session_scores),
                'improvement_trajectory': session_scores
            },
            'joint_performance': joint_scores
        }