"""
SHADOW SKILL - Advanced Tacit Knowledge Digitization System
Digitizes master craftsman expertise for apprentice training
"""

__version__ = "2.0-METI"
__author__ = "Shadow Skill Team"

from .pose_extractor import PoseExtractor
from .similarity_engine import SimilarityEngine
from .visualizer import Visualizer

__all__ = ["PoseExtractor", "SimilarityEngine", "Visualizer"]