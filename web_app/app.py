from flask import Flask, render_template, Response, jsonify
import cv2
import random

app = Flask(__name__)

# =====================================================
# GLOBALS
# =====================================================
cap = None
running = False

live_score = 0
feedback_msg = "Press Start"

# =====================================================
# VIDEO STREAM
# =====================================================
def generate_frames():

    global cap
    global running
    global live_score
    global feedback_msg

    while running:

        if cap is None:
            break

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        # =============================================
        # FAKE AI SCORE (TEMP PROTOTYPE)
        # =============================================
        live_score = random.randint(40, 100)

        if live_score > 85:
            feedback_msg = "Perfect Match 🔥"

        elif live_score > 65:
            feedback_msg = "Good Alignment"

        elif live_score > 40:
            feedback_msg = "Adjust Pose"

        else:
            feedback_msg = "Follow Master"

        # =============================================
        # UI
        # =============================================
        cv2.putText(
            frame,
            "SHADOW SKILL AI",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,255),
            2
        )

        cv2.putText(
            frame,
            f"Score: {live_score}%",
            (20,80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,255),
            2
        )

        cv2.putText(
            frame,
            feedback_msg,
            (20,120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,200,0),
            2
        )

        _, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )

# =====================================================
# ROUTES
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")

# START
@app.route("/start")
def start():

    global cap
    global running

    if not running:
        cap = cv2.VideoCapture(0)
        running = True

    return jsonify({"status":"started"})

# STOP
@app.route("/stop")
def stop():

    global cap
    global running

    running = False

    if cap is not None:
        cap.release()
        cap = None

    cv2.destroyAllWindows()

    return jsonify({"status":"stopped"})

# VIDEO
@app.route("/video")
def video():

    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

# SCORE API
@app.route("/score")
def score():

    return jsonify({
        "score": live_score,
        "feedback": feedback_msg
    })

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)