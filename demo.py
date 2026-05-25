#!/usr/bin/env python3
"""
Shadow Skill - Demo Script
Three modes: extract, compare, realtime
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.pose_extractor import PoseExtractor
from core.similarity_engine import SimilarityEngine
from core.visualizer import Visualizer

def demo_extract(video_path):
    """Mode 1: Extract pose sequence from video"""
    print("\n" + "="*60)
    print("🎯 MODE: EXTRACT POSE SEQUENCE")
    print("="*60)
    
    extractor = PoseExtractor()
    print(f"📹 Extracting from: {video_path}")
    frames = extractor.extract_sequence(video_path)
    
    print(f"✅ Extracted {len(frames)} frames")
    
    # Save
    output_path = f"data/extracted_{os.path.basename(video_path)}.json"
    extractor.save_sequence(frames, output_path)
    print(f"💾 Saved to: {output_path}")
    
    # Show first frame
    if frames:
        first_frame = frames[0]
        print(f"\n📊 First Frame Landmarks:")
        for joint, coords in list(first_frame['landmarks'].items())[:5]:
            print(f"  {joint}: x={coords['x']:.3f}, y={coords['y']:.3f}, z={coords['z']:.3f}")
        print(f"  ... and {len(first_frame['landmarks']) - 5} more joints")

def demo_compare(master_path, user_path):
    """Mode 2: Compare master vs user sequences"""
    print("\n" + "="*60)
    print("🎯 MODE: COMPARE SEQUENCES (Offline)")
    print("="*60)
    
    extractor = PoseExtractor()
    similarity_engine = SimilarityEngine()
    
    print(f"📹 Master: {master_path}")
    print(f"📹 User: {user_path}")
    
    master_frames = extractor.extract_sequence(master_path)
    user_frames = extractor.extract_sequence(user_path)
    
    print(f"✅ Master: {len(master_frames)} frames")
    print(f"✅ User: {len(user_frames)} frames")
    
    # Compare
    result = similarity_engine.compare_sequences(master_frames, user_frames)
    
    print(f"\n📊 RESULTS:")
    print(f"  Overall Score: {result.overall_score:.1f}%")
    print(f"  Spatial Score: {result.spatial_score:.1f}%")
    print(f"  Temporal Score: {result.temporal_score:.1f}%")
    print(f"  DTW Distance: {result.dtw_distance:.3f}")
    print(f"  Feedback: {result.feedback}")
    
    print(f"\n📊 Joint Scores:")
    for joint, score in sorted(result.joint_scores.items(), key=lambda x: x[1], reverse=True)[:6]:
        status = "✅" if score > 80 else "⚠️" if score > 60 else "❌"
        print(f"  {status} {joint}: {score:.1f}%")

def demo_realtime(master_path):
    """Mode 3: Real-time comparison (webcam)"""
    print("\n" + "="*60)
    print("🎯 MODE: REAL-TIME COMPARISON (Webcam)")
    print("="*60)
    print(f"📹 Master: {master_path}")
    print("📹 User: Webcam (press 'q' to quit)")
    
    extractor = PoseExtractor()
    similarity_engine = SimilarityEngine()
    
    master_frames = extractor.extract_sequence(master_path)
    visualizer = Visualizer(master_frames)
    
    print(f"✅ Master loaded: {len(master_frames)} frames")
    print(f"\n🎬 Starting real-time comparison...\n")
    
    session_scores = visualizer.render_real_time_session(
        user_video_path=None,  # Use webcam
        master_video_path=master_path,
        similarity_engine=similarity_engine
    )
    
    if session_scores:
        import statistics
        avg_score = statistics.mean(session_scores)
        print(f"\n📊 SESSION SUMMARY:")
        print(f"  Average Score: {avg_score:.1f}%")
        print(f"  Total Frames: {len(session_scores)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Shadow Skill Demo - Tacit Knowledge AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py --extract master_skill.mp4
  python demo.py --compare master.mp4 apprentice.mp4
  python demo.py --realtime master.mp4
        """
    )
    
    parser.add_argument('--extract', help='Extract pose sequence from video')
    parser.add_argument('--compare', nargs=2, metavar=('MASTER', 'USER'), help='Compare two videos')
    parser.add_argument('--realtime', help='Real-time webcam comparison against master')
    
    args = parser.parse_args()
    
    if args.extract:
        demo_extract(args.extract)
    elif args.compare:
        demo_compare(args.compare[0], args.compare[1])
    elif args.realtime:
        demo_realtime(args.realtime)
    else:
        parser.print_help()
        print("\n💡 TIP: Run one of the demo modes above!")