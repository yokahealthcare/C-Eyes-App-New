import cv2
import pygame
import time

# colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class FaceDetector:
    def __init__(self, frame, width):
        self.frame = frame
        self.width = width
        self.face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.faces = None
        self.faces_width = None
        self.faces_position = None

    def detect_faces(self):
        # Converting color image to grayscale image
        gray_image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # Detecting faces in the image
        self.faces = self.face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    def get_face_widths(self):
        self.faces_width = [w for _, _, w, _ in self.faces]

    def get_face_positions(self):
        positions = []
        for x, _, w, _ in self.faces:
            mx = x + int(w / 2)
            if mx > (self.width / 2):
                positions.append("R")
            else:
                positions.append("L")

        self.faces_position = positions

    def get_face_information(self):
        self.detect_faces()
        self.get_face_widths()
        self.get_face_positions()


class FaceDistanceEstimator():
    def __init__(self, FaceDetector):
        # get from references image
        self.real_distance = 70
        self.real_width = 15.5

        self.fd = FaceDetector
        self.focal_length = None
        self.distances = None

    def get_focal_length(self):
        # finding the focal length
        self.focal_length = (self.fd.faces_width * self.real_distance) / self.real_width

    def get_distance(self):
        self.distances = [(self.real_width * self.focal_length) / w for w in self.fd.faces_width]

    def get_face_distance_approx(self):
        data_dict = {"L": [], "R": []}
        for position, distance in zip(self.fd.faces_position, self.distances):
            data_dict[position].append(distance)

        return data_dict


if __name__ == "__main__":
    # read the references image
    ref_image = cv2.imread('ref_image.jpg')

    ref_fd = FaceDetector(ref_image, 0)
    # process face
    ref_fd.get_face_information()
    # error in detection, so the true face is at index 1
    ref_fd.faces_width = ref_fd.faces_width[1]

    # print(f"Faces                   : {ref_fd.faces}")
    # print(f"Faces Width             : {ref_fd.faces_width}")
    # print(f"Faces Position          : {ref_fd.faces_position}")

    # process distance
    ref_dist = FaceDistanceEstimator(ref_fd)
    ref_dist.get_focal_length()

    print(f"Focal Length            : {ref_dist.focal_length}")

    # Start the camera
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        fd = FaceDetector(frame, frame.shape[1])
        fd.get_face_information()

        # print(f"Faces                   : {fd.faces}")
        # print(f"Faces Width             : {fd.faces_width}")
        # print(f"Faces Position          : {fd.faces_position}")

        dist = FaceDistanceEstimator(fd)
        dist.focal_length = ref_dist.focal_length
        dist.get_distance()

        # print(f"Distance          : {dist.distances}")
        # print("Return Data Dict")
        # print(dist.get_face_distance_approx())

        distance_dict = dist.get_face_distance_approx()

        # initialize pygame
        pygame.init()
        pygame.mixer.init()

        # Get a free channel
        channel = pygame.mixer.find_channel()
        while channel is None:
            print("No free channels available. Waiting...")
            time.sleep(1)
            channel = pygame.mixer.find_channel()

        # Set the volume for left and right speakers
        left_volume = 0.0
        right_volume = 0.0
        # Set the volume for left and right speakers
        channel.set_volume(left_volume, right_volume)

        # Load and play the audio file
        sound = pygame.mixer.Sound('sound/alarm-fire.mp3')
        channel.play(sound, -1)

        # normalize the distance range 0 to 1
        MIN = 30  # centimeter
        MAX = 100  # centimeter

        if distance_dict["L"]:
            left = (min(distance_dict["L"]) - MIN) / (MAX - MIN)
            left_volume = 1.0 - left

        else:
            if left_volume > 0.0:
                # slowly decrease volume if no face detected
                left_volume -= 0.05

        if distance_dict["R"]:
            right = (min(distance_dict["R"]) - MIN) / (MAX - MIN)
            right_volume = 1.0 - right

        else:
            if right_volume > 0.0:
                # slowly decrease volume if no face detected
                right_volume -= 0.05

        # safety limit
        left_volume = max(0.0, min(left_volume, 1.0))
        right_volume = max(0.0, min(right_volume, 1.0))

        # Set the volume for left and right speakers
        channel.set_volume(left_volume, right_volume)

        # quit the program if you press 'q' on keyboard
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
