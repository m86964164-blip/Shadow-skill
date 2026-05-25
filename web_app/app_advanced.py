"""Flask Web App for Shadow Skill - Production Grade
Serves real-time pose comparison with ghost overlay
"""

from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import threading
import json
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pose_extractor import PoseExtractor
from core.similarity_engine import SimilarityEngine
from core.visualizer import Visualizer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# =====================================================
# GLOBALS & STATE MANAGEMENT
# =====================================================
class SessionState:
    def __init__(self):
        self.running = False
        self.cap = None
        self.master_frames = None
        self.live_score = 0
        self.feedback_msg = "Press Start"
        self.current_result = None
        self.session_scores = []
        self.lock = threading.Lock()

state = SessionState()
extractor = PoseExtractor()
similarity_engine = SimilarityEngine()
visualizer = None

# =====================================================
# VIDEO STREAM
# =====================================================
def generate_frames():
    """Generate video frames with pose overlay"""
    while state.running:
        if state.cap is None:
            break
        
        ret, frame = state.cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Extract user pose
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        
        if results.pose_landmarks and state.master_frames:
            # Convert landmarks
            user_landmarks = {}
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
            
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                user_landmarks[landmark_names[idx]] = {
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                    'visibility': float(landmark.visibility)
                }
            
            user_frame = {'landmarks': user_landmarks, 'frame_id': 0}
            comparison = similarity_engine.compare_sequences(state.master_frames[:1], [user_frame])
            
            with state.lock:
                state.live_score = comparison.overall_score
                state.feedback_msg = comparison.feedback
                state.current_result = comparison
                state.session_scores.append(state.live_score)
            
            # Render with overlay
            if visualizer:
                master_frame = state.master_frames[0]
                frame = visualizer.render_ghost_overlay(frame, master_frame, comparison)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame_bytes +
            b'\r\n'
        )

# =====================================================
# ROUTES
# =====================================================
@app.route('/')
def home():
    return render_template('index_advanced.html')

@app.route('/upload_master', methods=['POST'])
def upload_master():
    """Upload and process master video"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temp file
        temp_path = f"/tmp/{file.filename}"
        file.save(temp_path)
        
        # Extract pose sequence
        global visualizer
        state.master_frames = extractor.extract_sequence(temp_path)
        visualizer = Visualizer(state.master_frames)
        
        # Cleanup
        os.remove(temp_path)
        
        return jsonify({
            'status': 'success',
            'frames_extracted': len(state.master_frames),
            'message': f'✅ Master uploaded: {len(state.master_frames)} frames extracted'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start')
def start():
    """Start pose comparison session"""
    with state.lock:
        if not state.running:
            state.cap = cv2.VideoCapture(0)
            state.running = True
            state.session_scores = []
    
    return jsonify({'status': 'started'})

@app.route('/stop')
def stop():
    """Stop pose comparison session"""
    with state.lock:
        state.running = False
        if state.cap is not None:
            state.cap.release()
            state.cap = None
    
    cv2.destroyAllWindows()
    return jsonify({'status': 'stopped'})

@app.route('/video')
def video():
    """Stream video with pose overlay"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/score')
def score():
    """Get current score and feedback"""
    with state.lock:
        return jsonify({
            'score': state.live_score,
            'feedback': state.feedback_msg,
            'joint_scores': state.current_result.joint_scores if state.current_result else {}
        })

@app.route('/report')
def report():
    """Generate session report"""
    if visualizer and state.session_scores:
        report = visualizer.generate_session_report(
            state.session_scores,
            state.current_result.joint_scores if state.current_result else {}
        )
        return jsonify(report)
    return jsonify({'error': 'No session data'}), 400

@app.route('/export_report')
def export_report():
    """Export session report as JSON"""
    if visualizer and state.session_scores:
        report = visualizer.generate_session_report(
            state.session_scores,
            state.current_result.joint_scores if state.current_result else {}
        )
        return jsonify(report)
    return jsonify({'error': 'No session data'}), 400

# =====================================================
# RUN
# =====================================================
if __name__ == '__main__':
    print("🎎 Shadow Skill Web App Starting...")
    print("📍 http://localhost:5000")
    app.run(debug=True, threaded=True, use_reloader=False)