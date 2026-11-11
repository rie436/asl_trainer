import cv2, joblib
import numpy as np
import mediapipe as mp

model = joblib.load("asl_model.pkl")
mp_hands = mp.solutions.hands
draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def to_features(landmarks):
    base = landmarks[0]
    row = []
    for p in landmarks:
        row += [p.x - base.x, p.y - base.y, p.z - base.z]
    return row

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        features = np.array(to_features(hand.landmark)).reshape(1, -1)
        probs = model.predict_proba(features)[0]
        best = probs.argmax()
        letter = model.classes_[best]
        conf = probs[best]

        color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
        cv2.putText(frame, f"{letter}  {conf:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    cv2.imshow("ASL Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()