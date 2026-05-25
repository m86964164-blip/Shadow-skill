"""Similarity Engine with Dynamic Time Warping (DTW)
Compares user pose against master pose regardless of speed/tempo
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import math

@dataclass
class ComparisonResult:
    """Result of pose comparison"""
    overall_score: float
    spatial_score: float
    temporal_score: float
    joint_scores: Dict[str, float]
    feedback: str
    per_frame_scores: List[float]
    dtw_distance: float

class SimilarityEngine:
    """Compare poses using DTW and multi-factor scoring"""
    
    def __init__(self, spatial_weight=0.6, temporal_weight=0.4):
        """Initialize similarity engine"""
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.joint_names = [
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def compare_sequences(self, master_frames: List[Dict], user_frames: List[Dict]) -> ComparisonResult:
        """Compare user sequence against master using DTW"""
        
        # Extract coordinate sequences
        master_coords = self._extract_coordinates(master_frames)
        user_coords = self._extract_coordinates(user_frames)
        
        # Compute DTW distance and alignment
        dtw_distance, alignment = self._dtw(master_coords, user_coords)
        
        # Compute spatial score
        spatial_score = self._compute_spatial_score(master_coords, user_coords, alignment)
        
        # Compute temporal score (velocity consistency)
        temporal_score = self._compute_temporal_score(master_frames, user_frames)
        
        # Per-joint analysis
        joint_scores = self._compute_joint_scores(master_coords, user_coords, alignment)
        
        # Overall score
        overall_score = (self.spatial_weight * spatial_score) + (self.temporal_weight * temporal_score)
        overall_score = max(0, min(100, overall_score))  # Clamp 0-100
        
        # Generate feedback
        feedback = self._generate_feedback(overall_score, joint_scores)
        
        # Per-frame scores for visualization
        per_frame_scores = self._compute_per_frame_scores(master_coords, user_coords, alignment)
        
        return ComparisonResult(
            overall_score=overall_score,
            spatial_score=spatial_score,
            temporal_score=temporal_score,
            joint_scores=joint_scores,
            feedback=feedback,
            per_frame_scores=per_frame_scores,
            dtw_distance=dtw_distance
        )
    
    def _extract_coordinates(self, frames: List[Dict]) -> np.ndarray:
        """Extract (x, y, z) coordinates from frames"""
        coords = []
        for frame in frames:
            landmarks = frame['landmarks']
            frame_coords = []
            for joint in self.joint_names:
                if joint in landmarks:
                    lm = landmarks[joint]
                    frame_coords.extend([lm['x'], lm['y'], lm['z']])
            coords.append(frame_coords)
        return np.array(coords)
    
    def _dtw(self, master: np.ndarray, user: np.ndarray) -> Tuple[float, List]:
        """Dynamic Time Warping - aligns sequences of different lengths"""
        n, m = len(master), len(user)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Fill DTW matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = np.linalg.norm(master[i-1] - user[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # insertion
                    dtw_matrix[i, j-1],      # deletion
                    dtw_matrix[i-1, j-1]     # match
                )
        
        # Backtrack to find alignment
        alignment = []
        i, j = n, m
        while i > 0 or j > 0:
            alignment.append((i-1, j-1))
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                min_idx = np.argmin([dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]])
                if min_idx == 0:
                    i -= 1
                    j -= 1
                elif min_idx == 1:
                    i -= 1
                else:
                    j -= 1
        
        alignment.reverse()
        dtw_distance = dtw_matrix[n, m]
        
        return dtw_distance, alignment
    
    def _compute_spatial_score(self, master: np.ndarray, user: np.ndarray, alignment: List) -> float:
        """Compute spatial alignment score (0-100)"""
        distances = []
        for i, j in alignment:
            if i >= 0 and j >= 0 and i < len(master) and j < len(user):
                dist = np.linalg.norm(master[i] - user[j])
                distances.append(dist)
        
        mean_distance = np.mean(distances) if distances else 0
        # Convert distance to 0-100 score (lower distance = higher score)
        spatial_score = 100 * math.exp(-mean_distance * 2)
        return max(0, min(100, spatial_score))
    
    def _compute_temporal_score(self, master_frames: List[Dict], user_frames: List[Dict]) -> float:
        """Compute temporal consistency score"""
        master_length = len(master_frames)
        user_length = len(user_frames)
        
        # Penalize extreme speed differences
        length_ratio = max(master_length, user_length) / min(master_length, user_length)
        temporal_score = 100 / (1 + (length_ratio - 1) * 0.5)
        return max(0, min(100, temporal_score))
    
    def _compute_joint_scores(self, master: np.ndarray, user: np.ndarray, alignment: List) -> Dict[str, float]:
        """Compute per-joint similarity scores"""
        joint_scores = {name: [] for name in self.joint_names}
        
        for i, j in alignment:
            if i >= 0 and j >= 0 and i < len(master) and j < len(user):
                for idx, joint_name in enumerate(self.joint_names):
                    coord_idx = idx * 3
                    if coord_idx + 2 < len(master[i]):
                        master_joint = master[i][coord_idx:coord_idx+3]
                        user_joint = user[j][coord_idx:coord_idx+3]
                        dist = np.linalg.norm(master_joint - user_joint)
                        score = 100 * math.exp(-dist * 2)
                        joint_scores[joint_name].append(score)
        
        # Average scores per joint
        for joint in joint_scores:
            if joint_scores[joint]:
                joint_scores[joint] = max(0, min(100, np.mean(joint_scores[joint])))
            else:
                joint_scores[joint] = 0
        
        return joint_scores
    
    def _compute_per_frame_scores(self, master: np.ndarray, user: np.ndarray, alignment: List) -> List[float]:
        """Compute score for each user frame"""
        per_frame_scores = [0] * len(user)
        
        for i, j in alignment:
            if j >= 0 and j < len(user) and i >= 0 and i < len(master):
                dist = np.linalg.norm(master[i] - user[j])
                score = 100 * math.exp(-dist * 2)
                per_frame_scores[j] = max(per_frame_scores[j], score)
        
        return per_frame_scores
    
    def _generate_feedback(self, overall_score: float, joint_scores: Dict[str, float]) -> str:
        """Generate actionable feedback based on scores"""
        if overall_score > 85:
            feedback = "🔥 Perfect Match! Master-level precision!"
        elif overall_score > 70:
            feedback = "✅ Good Alignment - You're on track!"
        elif overall_score > 50:
            feedback = "⚠️ Adjust your posture and timing"
        else:
            feedback = "📖 Follow the master more carefully"
        
        # Add specific joint feedback
        worst_joints = sorted(joint_scores.items(), key=lambda x: x[1])[:2]
        if worst_joints and worst_joints[0][1] < 60:
            joint_names = [j.replace('_', ' ').title() for j, _ in worst_joints]
            feedback += f" - Fix: {', '.join(joint_names)}"
        
        return feedback