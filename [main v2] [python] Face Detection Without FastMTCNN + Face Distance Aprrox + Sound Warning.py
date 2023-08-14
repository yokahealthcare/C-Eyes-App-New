from facenet_pytorch import MTCNN
from PIL import Image
import torch
import cv2
import numpy as np
import pygame
import os

# colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

# get from references image
known_distance = 70
known_width = 15.5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device being used : {device}")

# Create face detector
mtcnn = MTCNN(device=device)

def get_width_pixel(frame):

    face_width = [] # making face width to zero

    # Detect face
    boxes, probs = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            # convert to int
            box = [int(x) for x in box]
            x_l, y_l, x_r, y_r = box
            cv2.rectangle(frame, (x_l, y_l), (x_r, y_r), GREEN, 2)

            face_width.append(x_r - x_l)

    # return the face width in pixel
    return face_width

# focal length finder function
def get_focal_length(real_distance, real_width, width_in_rf_image):
    # finding the focal length
    focal_length = (width_in_rf_image * real_distance) / real_width
    return focal_length

# distance estimation function
def get_distance(focal_length, real_width, width_in_rf_image):
    distance = []
    for w in width_in_rf_image:
        distance.append((real_width * focal_length) / w)

    # return the distance
    return distance

def get_face_distance_approx(focal_length, known_width, frame):
    distance = []
    # get the width of pixel from the image
    test_face_width_pixel = get_width_pixel(frame)

    if test_face_width_pixel != 0:
        # get distance
        distance = get_distance(focal_length, known_width, test_face_width_pixel)

        # Drawing Text on the screen
        for idx,d in enumerate(distance):
            cv2.putText(frame, f"Person {idx+1}: {round(d,2)} CM", (30, 35+idx*30), fonts, 0.6, GREEN, 1)

    return frame, distance



def main():
    # initialize pygame
    pygame.init()
    pygame.mixer.init()

    # Load and play the music file
    pygame.mixer.music.load('sound/bluestone-alley.mp3')
    pygame.mixer.music.play(-1)  # -1 means play the music on loop

    # Set the initial volume to 0.0 (0% volume)
    volume = 0.0
    pygame.mixer.music.set_volume(volume)


    # read the references image
    ref_image = cv2.imread('ref_image.jpg')
    # getting the pixel width of the reference image first
    ref_face_width_pixel = get_width_pixel(ref_image)
    print(f"ref face : {ref_face_width_pixel}")
    # calculate the focal length
    # the index 1, because it detect two faces arrogantly, the correct one at index 1
    focal_length = get_focal_length(known_distance, known_width, ref_face_width_pixel[0])
    print(f"Focal Length has been Founded \t: {focal_length}\n")

    cap = cv2.VideoCapture(0)

    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    while cap.isOpened():

        success, frame = cap.read()
        if not success:
            print("Video has ended!")
            break

        frame, distance = get_face_distance_approx(focal_length, known_width, frame)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()