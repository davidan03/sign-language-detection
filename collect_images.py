import os
import cv2

DATA_DIR = "./data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

num_classes = 24
dataset_size = 100

capture = cv2.VideoCapture(0)

for i in range(num_classes):
    if not os.path.exists(os.path.join(DATA_DIR, str(i))):
        os.makedirs(os.path.join(DATA_DIR, str(i)))

    print(f"Collecting data for class {str(i)}")

    while True:
        ret, frame = capture.read()
        cv2.putText(frame, 'Ready? Press "SPACE" !!!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3)
        cv2.imshow("Data Collection", frame)

        if cv2.waitKey(25) & 0xFF == ord(' '):
            break

    counter = 0

    while counter < dataset_size:
        ret, frame = capture.read()
        cv2.imshow("Data Collection", frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DATA_DIR, str(i), f"{counter}.jpg"), frame)

        counter += 1

capture.release()
cv2.destroyAllWindows()