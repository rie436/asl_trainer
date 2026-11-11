import cv2, joblib, random, time
import numpy as np
import mediapipe as mp

model = joblib.load("asl_model.pkl")
letters = list(model.classes_)

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

target = random.choice(letters)
score = 0
hold_start = None
HOLD_SECONDS = 1.0
CONFIDENCE = 0.7

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    guess, conf = None, 0
    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        feats = np.array(to_features(hand.landmark)).reshape(1, -1)
        probs = model.predict_proba(feats)[0]
        guess, conf = model.classes_[probs.argmax()], probs.max()

    if guess == target and conf > CONFIDENCE:
        if hold_start is None:
            hold_start = time.time()
        held = time.time() - hold_start
        if held >= HOLD_SECONDS:
            score += 1
            target = random.choice([l for l in letters if l != target])
            hold_start = None
        status, color = f"Correct! Hold it... {held:.1f}s", (0, 255, 0)
    else:
        hold_start = None
        status = f"You're showing: {guess}" if guess else "Show your hand"
        color = (0, 165, 255)

    cv2.putText(frame, f"Sign the letter: {target}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    cv2.putText(frame, status, (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Score: {score}", (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("ASL Practice", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()