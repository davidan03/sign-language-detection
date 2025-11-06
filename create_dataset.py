import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = "./data"

data = []
labels = []

for dir in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir)):
        if not img_path.endswith(".jpg"):
            continue

        landmark_data = []

        x_coords = []
        y_coords = []

        img = cv2.imread(os.path.join(DATA_DIR, dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                for i in range(len(hand.landmark)):
                    x = hand.landmark[i].x
                    y = hand.landmark[i].y

                    x_coords.append(x)
                    y_coords.append(y)

                    landmark_data.append(x - min(x_coords))
                    landmark_data.append(y - min(y_coords))

            data.append(landmark_data)
            labels.append(dir)

f = open('data.pickle', 'wb')
pickle.dump({"data": data, "labels": labels}, f)
f.close()