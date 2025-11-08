import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open("./model.pickle", "rb"))
model = model_dict["model"]
labels_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G",
               7: "H", 8: "I", 9: "K", 10: "L", 11: "M", 12: "N",
               13: "O", 14: "P", 15: "Q", 16: "R", 17: "S", 18: "T",
               19: "U", 20: "V", 21: "W", 22: "X", 23: "Y"}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

capture = cv2.VideoCapture(0)
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

while True:
    landmark_data = []

    x_coords = []
    y_coords = []

    ret, frame = capture.read()

    if not ret:
        continue

    H, W, C = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        
        for i in range(len(hand.landmark)):
            x = hand.landmark[i].x
            y = hand.landmark[i].y

            x_coords.append(x)
            y_coords.append(y)

            landmark_data.append(x - min(x_coords))
            landmark_data.append(y - min(y_coords))

        mp_drawing.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        x1 = int(min(x_coords) * W) - 10
        y1 = int(min(y_coords) * H) - 10

        x2 = int(max(x_coords) * W) - 10
        y2 = int(max(y_coords) * H) - 10

        pred = model.predict([np.asarray(landmark_data)])
        predicted_character = labels_dict[int(pred[0])]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow("Sign Language Detection", frame)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()