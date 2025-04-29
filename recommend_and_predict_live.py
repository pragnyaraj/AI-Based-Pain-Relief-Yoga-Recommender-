import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load pose classifier
pose_model = joblib.load("pose_classifier.pkl")
pose_names = ['downdog', 'goddess', 'plank', 'tree', 'warrior2']

# Load user pain recommendation model
user_data = pd.read_csv(r"C:\Users\pragn\YogaPoseRecommender\user_input_log.csv")  # should include 'Pain_Area', 'Pain_Intensity', 'Pose'
X = user_data[['Pain Area', 'Pain Intensity','Pose']]

X = pd.get_dummies(X)
y = user_data['Pose']

recommendation_model = DecisionTreeClassifier()
recommendation_model.fit(X, y)

# Get user input
pain_area = input("Enter pain area (e.g., neck, back, knee): ").lower()
pain_intensity = int(input("Enter pain intensity (1 to 10): "))

# Prepare input for recommendation
input_df = pd.DataFrame([{'Pain Area': pain_area, 'Pain Intensity': pain_intensity}])


input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Recommend pose
recommended_pose = recommendation_model.predict(input_df)[0]
print(f"\n🧘 Recommended Yoga Pose: {recommended_pose.upper()}")

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Normalize landmarks
def normalize_landmarks(landmarks):
    if landmarks.shape != (33, 3):
        return None
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    center = (left_hip + right_hip) / 2
    normalized = landmarks - center
    max_val = np.max(np.abs(normalized))
    if max_val != 0:
        normalized /= max_val
    return normalized.flatten()

# Open camera
cap = cv2.VideoCapture(0)
print("\n Camera started — Perform the recommended pose")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    predicted_pose = "No pose"
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
        normalized = normalize_landmarks(landmarks)

        if normalized is not None:
            probs = pose_model.predict_proba([normalized])[0]
            max_prob = np.max(probs)
            label = pose_model.predict([normalized])[0]

            if max_prob > 0.8:
                predicted_pose = label
            else:
                predicted_pose = "Uncertain"

            print(f"Prediction: {label}, Confidence: {max_prob:.2f}")

    color = (0, 255, 0) if predicted_pose == recommended_pose else (0, 0, 255)
    cv2.putText(frame, f"Pose: {predicted_pose}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Target: {recommended_pose}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Yoga Pose Guidance', frame)
    if cv2.waitKey(100) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
