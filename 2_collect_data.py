import cv2, csv, sys
import mediapipe as mp

label = sys.argv[1].upper()

mp_hands = mp.solutions.hands
draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
count = 0

def to_features(landmarks):
    base = landmarks[0]
    row = []
    for p in landmarks:
        row += [p.x - base.x, p.y - base.y, p.z - base.z]
    return row

with open("asl_data.csv", "a", newline="") as f:
    writer = csv.writer(f)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_found = bool(result.multi_hand_landmarks)

        if hand_found:
            hand = result.multi_hand_landmarks[0]
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and hand_found:
            writer.writerow([label] + to_features(hand.landmark))
            count += 1
        elif key == ord("q"):
            break

        cv2.putText(frame, f"Letter {label}: {count} saved  (S=save, Q=quit)",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collect", frame)

cap.release()
cv2.destroyAllWindows()