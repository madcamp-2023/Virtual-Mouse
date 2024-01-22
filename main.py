import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

drawing = False
img = None

def draw(event, x, y, flags, param):
    global drawing, img
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img, (x,y), 5, (255, 0, 0), -1)
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

cv2.namedWindow('Hand Tracking')
cv2.setMouseCallback('Hand Tracking', draw)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                if landmark == hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]:
                    if img is None:
                        img = 255 * np.ones_like(frame)

                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)

    cv2.imshow('Hand Tracking', frame)

    if img is not None:
        cv2.imshow('Drawing', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elif cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwirte('drawn_image.png', img)
        print("Drawing saved as 'drawn_image.png'")

cap.release()
cv2.destroyAllWindows()
