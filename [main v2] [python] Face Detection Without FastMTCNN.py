from facenet_pytorch import MTCNN
from PIL import Image
import torch
import cv2
import numpy as np

# colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device being used : {device}")

# Create face detector
mtcnn = MTCNN( device=device)

cap = cv2.VideoCapture(0)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():

    success, ori_frame = cap.read()
    if not success:
        print("Video has ended!")
        break

    frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # Detect face
    boxes, probs = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            #convert to int
            box = [int(x) for x in box]
            x_l, y_l, x_r, y_r = box
            cv2.rectangle(ori_frame, (x_l, y_l), (x_r, y_r), GREEN, 2)

    cv2.imshow("frame", ori_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()