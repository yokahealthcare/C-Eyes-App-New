# library
import cv2

# face detector object
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

# colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# get from references image
known_distance = 70
known_width = 15.5

# width of the video feed
WIDTH = 0


def detect_faces(frame):
    # converting color image to gray scale image
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detecting face in the image
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    return faces


def get_width_pixel(faces):
    face_width = []  # making face width to zero

    for x, y, w, h in faces:
        # getting face width in the pixels
        face_width.append(w)

    # return the face width in pixel
    return face_width


def get_face_position(faces):
    position = []
    for x, y, w, h in faces:
        mx = x + int(w / 2)
        # print(f"mx : {mx}")
        # print(f"WIDTH : {WIDTH}")

        if mx > (WIDTH / 2):
            position.append("R")
        else:
            position.append("L")

    return position


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


def get_face_distance_approx(focal_length, known_width, faces):
    distance = []
    # get the width of pixel from the image
    test_face_width_pixel = get_width_pixel(faces)

    if test_face_width_pixel != 0:
        # get face position
        face_position = get_face_position(faces)
        # print(f"Face Position : {face_position}")
        # get distance
        distance = get_distance(focal_length, known_width, test_face_width_pixel)

    data_list = zip(face_position, distance)

    data_dict = {"L": [], "R": []}
    for position, distance in data_list:
        data_dict[position].append(distance)

    return data_dict


def retrieve_focal_length():
    # read the references image
    ref_image = cv2.imread('ref_image.jpg')
    # detect faces on references image
    faces = detect_faces(ref_image)
    # getting the pixel width of the reference image first
    ref_face_width_pixel = get_width_pixel(faces)
    # calculate the focal length
    # the index 1, because it detect two faces arrogantly, the correct one at index 1
    focal_length = get_focal_length(known_distance, known_width, ref_face_width_pixel[1])
    print(f"Focal Length has been Founded \t: {focal_length}\n")

    return focal_length


# focal length
focal_length = retrieve_focal_length()
