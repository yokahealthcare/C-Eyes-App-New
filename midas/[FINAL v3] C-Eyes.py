# [FINAL v3] C-Eyes.py
# Added Features:
# 1. ...
# Problem:
# 1. ...

import cv2
import pygame
import time
from mtcnn_cv2 import MTCNN
import torch

import numpy as np
from midas.model_loader import load_model

# colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX


class FaceDetector:
    def __init__(self):
        self.frame = None
        self.width = None
        self.face_detector = MTCNN()
        self.faces = None
        self.faces_width = None
        self.faces_position = None

    def reset(self):
        self.frame = None
        self.width = None
        self.faces = None
        self.faces_width = None
        self.faces_position = None

    def set_frame(self, f):
        self.frame = f
        self.width = f.shape[1]

    def detect_faces(self):
        # Converting color image to grayscale image
        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        # Detecting faces in the image
        self.faces = self.face_detector.detect_faces(image)
        # Extract only the 'box' values into a 2D array
        self.faces = [entry['box'] for entry in self.faces]

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

    def debug(self):
        print(f"Faces                   : {self.faces}")
        print(f"Faces Width             : {self.faces_width}")
        print(f"Faces Position          : {self.faces_position}")


class FaceDistanceEstimator:
    def __init__(self, FaceDetectorObject):
        # get from references image
        self.real_distance = 70
        self.real_width = 15.5

        self.fd = FaceDetectorObject
        self.focal_length = None
        self.distances = None

    def reset(self):
        self.distances = None

    def get_focal_length(self):
        # finding the focal length
        self.focal_length = (self.fd.faces_width[0] * self.real_distance) / self.real_width

    def get_distance(self):
        self.distances = [(self.real_width * self.focal_length) / w for w in self.fd.faces_width]

    def get_face_distance_approx(self):
        data_dict = {"L": [], "R": []}
        for position, distance in zip(self.fd.faces_position, self.distances):
            data_dict[position].append(distance)

        return data_dict

    def debug(self):
        print(f"Distance        : {self.distances}")


class SurroundAudio:
    def __init__(self):
        # Normalize the distance range 0 to 1
        self.MIN = 30  # centimeter
        self.MAX = 100  # centimeter

        # Get distance dictionary
        self.distance_dict = None

        # Get a free channel
        self.channel = pygame.mixer.find_channel()
        while self.channel is None:
            print("No free channels available. Waiting...")
            time.sleep(1)
            self.channel = pygame.mixer.find_channel()

        # Set the volume for left and right speakers
        self.left_volume = 0.0
        self.right_volume = 0.0

        # Set the volume for left and right speakers
        self.channel.set_volume(self.left_volume, self.right_volume)

        # Load the default audio file
        self.audio = pygame.mixer.Sound("../sound/alarm-fire.mp3")

    def load_audio(self, audiofile):
        # Load the audio file
        self.audio = pygame.mixer.Sound(audiofile)

    def play_audio(self):
        self.channel.play(self.audio, -1)

    def set_volume(self, left_vol, right_vol):
        # safety limit
        self.left_volume = max(0.0, min(left_vol, 1.0))
        self.right_volume = max(0.0, min(right_vol, 1.0))

        # Set the volume for left and right speakers
        self.channel.set_volume(self.left_volume, self.right_volume)

    def compute(self, dist_dict):
        self.distance_dict = dist_dict

        if self.distance_dict["L"]:
            left = (min(self.distance_dict["L"]) - self.MIN) / (self.MAX - self.MIN)
            self.left_volume = 1.0 - left
        else:
            if self.left_volume > 0.0:
                # slowly decrease volume if no face detected
                self.left_volume -= 0.05

        if self.distance_dict["R"]:
            right = (min(self.distance_dict["R"]) - self.MIN) / (self.MAX - self.MIN)
            self.right_volume = 1.0 - right

        else:
            if self.right_volume > 0.0:
                # slowly decrease volume if no face detected
                self.right_volume -= 0.05

        self.set_volume(self.left_volume, self.right_volume)


class ImageController:
    def __init__(self, f):
        self.frame = f
        self.faces = None
        self.distance = None

    def annotate(self):
        if self.faces is None or self.distance is None:
            print(f"You haven't set the 'faces' and 'distance'")
            return 0

        # Box
        for x, y, w, h in self.faces:
            # draw the rectangle on the face
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), GREEN, 2)

        # Text
        for idx, d in enumerate(self.distance):
            cv2.putText(self.frame, f"Person {idx + 1}: {round(d, 2)} CM", (30, 35 + idx * 20), fonts, 0.5, GREEN, 1)

    def show(self, title="Showing Image"):
        cv2.imshow(title, self.frame)

    def set_faces(self, f):
        self.faces = f

    def set_distance(self, d):
        self.distance = d


class Midas:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = "weights/midas_v21_small_256.pt"
        self.model_type = "midas_v21_small_256"
        self.optimize = False
        self.height = None
        self.square = False
        self.grayscale = False
        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, self.model_path, self.model_type,
                                                                        self.optimize, self.height, self.square)
        self.prediction = None
        self.faces = None
        self.midas_area = None

        input_size = (self.net_w, self.net_h)
        print(f"Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")

    def get_prediction(self, frame):
        if frame is not None:
            original_image_rgb = np.flip(frame, 2)  # in [0, 255] (flip required to get RGB)
            image = self.transform({"image": original_image_rgb / 255})["image"]

            sample = torch.from_numpy(image).to(self.device).unsqueeze(0)
            target_size = original_image_rgb.shape[1::-1]

            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=target_size[::-1],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            # Calculate inverse depth
            # inverse_depth = 1.0 / prediction

            # Normalize the prediction
            self.prediction = self.normalize(prediction)

    def normalize(self, depth):
        depth_min = depth.min()
        depth_max = depth.max()
        normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
        normalized_depth *= 3

        result = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
        if not self.grayscale:
            result = cv2.applyColorMap(np.uint8(result), cv2.COLORMAP_INFERNO)

        return result

    def set_faces(self, faces):
        self.faces = faces

    def cut_midas_area(self):
        if self.faces is not None:
            midas_cut = []
            num_faces = 0
            for x, y, w, h in self.faces:
                num_faces += 1
                midas_cut.append(self.prediction[y:y + h, x:x + w])

            self.midas_area = midas_cut

    def show_midas_area(self):
        for idx, area in enumerate(self.midas_area):
            cv2.imshow(f"midas area {idx + 1}", area)

    def debug(self):
        print("\n\n")
        print(f"Prediction Shape: {self.prediction.shape}")
        print(f"Midas Area : ")
        for idx, area in enumerate(self.midas_area):
            print(f"Faces {idx + 1} : {area.shape}")
        print()
        print(f"Faces : {self.faces}")


def main():
    # Read the references image
    ref_image = cv2.imread('../ref_image.jpg')

    # Process face detector on reference image
    ref_fd = FaceDetector()
    ref_fd.set_frame(ref_image)
    ref_fd.get_face_information()

    # Process distance on reference image
    ref_dist = FaceDistanceEstimator(ref_fd)
    ref_dist.get_focal_length()

    # initialize pygame
    pygame.init()
    pygame.mixer.init()

    # Define SurroundAudio object
    speaker = SurroundAudio()
    speaker.play_audio()

    # Define FaceDetector
    fd = FaceDetector()

    # Create object of FaceDistanceEstimator
    dist = FaceDistanceEstimator(fd)
    # Set the focal length value to object
    dist.focal_length = ref_dist.focal_length

    # Define the Midas
    mi = Midas()

    with torch.no_grad():
        fps = 1
        # Start the camera
        cap = cv2.VideoCapture(0)
        time_start = time.time()

        while True:
            ret, frame = cap.read()

            # Midas
            mi.get_prediction(frame)

            # Set frame of FaceDetector & Get information about the face
            fd.set_frame(frame)
            fd.get_face_information()

            # Midas' calculation
            # Setting faces coordinate
            mi.set_faces(fd.faces)
            mi.cut_midas_area()
            mi.debug()

            # Calculating distance with FaceDistanceEstimator
            # Get distance of face
            dist.get_distance()

            # Create object of Image Controller
            im_controller = ImageController(f=frame)
            # Set the 'faces' inside the ImageController object
            im_controller.set_faces(fd.faces)
            # Set the 'distance' inside the ImageController object
            im_controller.set_distance(dist.distances)
            # Process & Show the frame
            im_controller.annotate()
            im_controller.show()

            # Get the distance value in form of dictionary
            distance_dict = dist.get_face_distance_approx()

            # Compute the distance with 8D audio
            speaker.compute(distance_dict)

            # FPS measurement
            alpha = 0.1
            if time.time() - time_start > 0:
                fps = (1 - alpha) * fps + alpha * 1 / (time.time() - time_start)  # exponential moving average
                time_start = time.time()
            print(f"\rFPS: {round(fps, 2)}", end="")

            # Reset the FaceDetector attributes
            fd.reset()
            # Reset the FaceDistanceEstimator attributes
            dist.reset()

            # quit the program if you press 'q' on keyboard
            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
