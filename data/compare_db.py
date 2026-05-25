import cv2
import mediapipe as mp
import numpy as np
import csv

# ---------------- LOAD MASTER DATA ----------------
master_data = []

with open("data/master.csv", "r") as f:
    reader = csv.reader(f)
    for row in reader:
        master_data.append(eval(row[0]))


# ---------------- MEDIAPIPE SETUP ----------------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE
)

detector = PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

frame_index = 0


# ---------------- DISTANCE FUNCTION ----------------
def distance(user, master):
    user = np.array(user)
    master = np.array(master)
    return np.mean(np.linalg.norm(user - master, axis=1))


# ---------------- LOOP ----------------
print("Comparing with MASTER... Press Q to stop")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.pose_landmarks and frame_index < len(master_data):

        lm = result.pose_landmarks[0]

        shoulder = [lm[11].x, lm[11].y]
        elbow = [lm[13].x, lm[13].y]
        wrist = [lm[15].x, lm[15].y]

        user_points = [shoulder, elbow, wrist]
        master_points = master_data[frame_index]

        diff = distance(user_points, master_points)

        score = max(0, 100 - diff * 200)

        # Feedback
        if score > 90:
            text = "Perfect"
        elif score > 75:
            text = "Good"
        else:
            text = "Fix posture"

        cv2.putText(frame, f"Score: {int(score)}%", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, text, (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_index += 1

    cv2.imshow("Shadow Skill - Compare Mode", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()