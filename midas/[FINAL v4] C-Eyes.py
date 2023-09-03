# [FINAL v4] C-Eyes.py
# Added Features:
# 1. Added Object Detection YOLOv5
# Problem:
# 1. All Good

import cv2
import pygame
import time
from mtcnn_cv2 import MTCNN
import torch
import os

import numpy as np
from midas.model_loader import load_model

import pyfiglet
import yaml

# colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# defining the fonts
fonts = cv2.FONT_HERSHEY_COMPLEX

# initialize pygame
pygame.init()
pygame.mixer.init()


def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


class FaceDetector:
    def __init__(self):
        self.frame = None
        self.width = None
        self.face_detector = MTCNN()
        self.faces = None
        self.widths = None
        self.positions = None

    def reset(self):
        self.frame = None
        self.width = None
        self.faces = None
        self.widths = None
        self.positions = None

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
        self.widths = [w for _, _, w, _ in self.faces]

    def get_face_positions(self):
        positions = []
        for x, _, w, _ in self.faces:
            mx = x + int(w / 2)
            if mx > (self.width / 2):
                positions.append("R")
            else:
                positions.append("L")

        self.positions = positions

    def get_face_information(self):
        self.detect_faces()
        self.get_face_widths()
        self.get_face_positions()

    def debug(self):
        print(f"Faces                   : {self.faces}")
        print(f"Faces Width             : {self.widths}")
        print(f"Faces Position          : {self.positions}")


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
        self.focal_length = (self.fd.widths[0] * self.real_distance) / self.real_width

    def get_distance(self):
        self.distances = [(self.real_width * self.focal_length) / w for w in self.fd.widths]

    def get_face_distance_approx(self):
        data_dict = {"L": [], "R": []}
        for position, distance in zip(self.fd.positions, self.distances):
            data_dict[position].append(distance)

        return data_dict

    def debug(self):
        print(f"Distance        : {self.distances}")


class ObjectDetector:
    def __init__(self):
        # Model
        self.model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom
        self.frame = None
        self.width = None
        self.results = None
        self.predictions = None
        self.boxes = None
        self.scores = None
        self.categories = None
        self.widths = None
        self.positions = None
        self.distances = None

    def get_object_information(self):
        if self.frame is None:
            print(f"You haven't set the frame")
            return 0

        # perform inference
        self.results = self.model(self.frame)

        # parse results
        self.predictions = self.results.pred[0]
        # convert boxes from Tensor to List
        self.boxes = self.predictions[:, :4].cpu().numpy()  # x1, y1, x2, y2
        # Use NumPy to find indices where categories equal 0
        zero_idx = np.where(self.predictions[:, 5].cpu() == 0.0)[0]

        # Filter out the 'person' categories from self.categories and self.boxes
        self.categories = [int(i) for i in self.predictions[:, 5] if i != 0]
        self.boxes = np.delete(self.boxes, zero_idx, axis=0)
        # Extract other information
        self.scores = self.predictions[:, 4]

        # Calculate object_width using NumPy
        self.widths = self.boxes[:, 2] - self.boxes[:, 0]

        # Update the width of video capture
        self.width = self.frame.shape[1]
        # Getting objects position
        self.get_object_positions()

    def set_frame(self, f):
        self.frame = f

    def set_model(self, m):
        self.model = torch.hub.load("ultralytics/yolov5", m)

    def get_object_positions(self):
        positions = []
        for x, _, w, _ in self.boxes:
            mx = x + int(w / 2)
            if mx > (self.width / 2):
                positions.append("R")
            else:
                positions.append("L")

        self.positions = positions

    def reset(self):
        self.frame = None
        self.width = None
        self.results = None
        self.predictions = None
        self.boxes = None
        self.scores = None
        self.categories = None
        self.widths = None
        self.positions = None
        self.distances = None

    def debug(self):
        print(f"Distance        : {self.distances}")


class SurroundAudio:
    def __init__(self):
        # Set MIN and MAX to Normalize the distance range 0 to 1
        self.MIN = None
        self.MAX = None

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
        self.audio = pygame.mixer.Sound("../sound/object-ding.mp3")

    def set_audio(self, audiofile):
        self.audio = pygame.mixer.Sound(audiofile)

    def play_audio(self):
        self.channel.play(self.audio, -1)

    def stop_audio(self):
        self.channel.stop()

    def set_volume(self, left_vol, right_vol):
        # safety limit
        self.left_volume = max(0.0, min(left_vol, 1.0))
        self.right_volume = max(0.0, min(right_vol, 1.0))

        # Set the volume for left and right speakers
        self.channel.set_volume(self.left_volume, self.right_volume)

    def set_min_max(self, minimum, maximum):
        self.MIN = minimum
        self.MAX = maximum

    def compute(self, dist_dict):
        if self.MIN is None or self.MAX is None:
            print(f"You haven't set the 'MIN' and 'MAX'")
            return 0

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
        self.mean_area = None
        self.objects_box = None
        self.objects_distance = None
        self.objects_categories = None
        self.yaml_objects = yaml_load("coco128.yaml")['names']

    def annotate(self, t):
        if t == "normal":
            if self.faces is None or self.distance is None:
                print(f"You haven't set the 'faces' and 'distance'")
                return 0

            # Box Faces
            for idx, (x, y, w, h) in enumerate(self.faces):
                # draw the rectangle on the face
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), RED, 2)
                # draw the text on the face
                text_size = max(int(0.03 * w), 1)  # Adjust the factor as needed
                cv2.putText(self.frame, f"Person {idx + 1}: {round(self.distance[idx], 2)} CM", (x + 10, y + 20),
                            fonts, text_size / 10, RED, 1)

            # Box Object
            if self.objects_box is not None and self.objects_distance:
                for idx, (x1, y1, x2, y2) in enumerate(self.objects_box):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # draw the rectangle on the object
                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), GREEN, 2)
                    # draw the text on the object
                    w = x2 - x1
                    text_size = max(int(0.03 * w), 1)  # Adjust the factor as needed
                    cv2.putText(self.frame,
                                f"{self.yaml_objects[self.objects_categories[idx]]} {idx + 1}: {round(self.objects_distance[idx], 2)} CM",
                                (x1 + 10, y1 + 20),
                                fonts, text_size / 10, GREEN, 1)

        elif t == "depthmap":
            if self.faces is None or self.mean_area is None:
                print(f"You haven't set the 'faces' and 'mean_area'")
                return 0

            # Box Faces
            for idx, (x, y, w, h) in enumerate(self.faces):
                # draw the rectangle on the face
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), RED, 2)
                # draw the text on the face
                text_size = max(int(0.03 * w), 1)  # Adjust the factor as needed
                cv2.putText(self.frame, f"Person {idx + 1}: {round(self.mean_area[idx], 2)}", (x + 10, y + 20),
                            fonts, text_size / 10, RED, 1)

            # Box Object
            if self.objects_box is not None and self.objects_distance:
                for idx, (x1, y1, x2, y2) in enumerate(self.objects_box):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    # draw the rectangle on the object
                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), GREEN, 2)
                    # draw the text on the object
                    w = x2 - x1
                    text_size = max(int(0.03 * w), 1)  # Adjust the factor as needed
                    cv2.putText(self.frame,
                                f"{self.yaml_objects[self.objects_categories[idx]]} {idx + 1}: {round(self.objects_distance[idx], 2)}",
                                (x1 + 10, y1 + 20),
                                fonts, text_size / 10, GREEN, 1)



    def show(self, title="Showing Image"):
        cv2.imshow(title, self.frame)

    def set_faces(self, f):
        self.faces = f

    def set_distance(self, d):
        self.distance = d

    def set_mean_area(self, m):
        self.mean_area = m

    def set_objects_box(self, o):
        self.objects_box = o

    def set_objects_distance(self, o):
        self.objects_distance = o

    def set_objects_categories(self, o):
        self.objects_categories = o


class Midas:
    def __init__(self, model_type="midas_v21_small_256"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = f"weights/{model_type}.pt"
        self.model_type = model_type
        self.optimize = False
        self.height = None
        self.square = False
        self.grayscale = False
        self.model, self.transform, self.net_w, self.net_h = load_model(self.device, self.model_path, self.model_type,
                                                                        self.optimize, self.height, self.square)
        self.prediction = None
        self.faces = None
        self.faces_position = None
        self.object_boxes = None
        self.objects_position = None
        self.midas_area = {"face": None, "object": None}
        self.mean_areas = None

        input_size = (self.net_w, self.net_h)
        print(f"Input resized to {input_size[0]}x{input_size[1]} before entering the encoder")

    def reset(self):
        self.prediction = None
        self.faces = None
        self.faces_position = None
        self.object_boxes = None
        self.objects_position = None
        self.midas_area = {"face": None, "object": None}
        self.mean_areas = None

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

    def set_faces_position(self, faces_position):
        self.faces_position = faces_position

    def set_objects_position(self, objects_position):
        self.objects_position = objects_position

    def set_objects_boxes(self, objects_boxes):
        self.object_boxes = objects_boxes

    def cut_midas_area(self):
        if self.faces is not None:
            midas_cut = []
            for x, y, w, h in self.faces:
                midas_cut.append(self.prediction[y:y + h, x:x + w])

            self.midas_area["face"] = midas_cut

        if self.object_boxes is not None:
            midas_cut = []
            for x1, y1, x2, y2 in self.object_boxes:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                midas_cut.append(self.prediction[y1:y2, x1:x2])

            self.midas_area["object"] = midas_cut

    def calculate_mean_area(self):
        if self.midas_area is not None:
            mean_areas = {}

            # Calculate mean depth value for "face"
            if "face" in self.midas_area:
                face_areas = self.midas_area["face"]
                face_mean_depths = [200 - np.mean(area) for area in face_areas]

                mean_areas["face"] = face_mean_depths

            # Calculate mean depth value for "object"
            if "object" in self.midas_area:
                object_areas = self.midas_area["object"]
                object_mean_depths = [200 - np.mean(area) for area in object_areas]
                mean_areas["object"] = object_mean_depths

            self.mean_areas = mean_areas

    def show_midas_area(self):
        for idx, area in enumerate(self.midas_area):
            cv2.imshow(f"midas area {idx + 1}", area)

    def get_face_distance_approx(self, t):
        data_dict = {"L": [], "R": []}
        if t == "face":
            for position, distance in zip(self.faces_position, self.mean_areas["face"]):
                data_dict[position].append(distance)
        else:
            for position, distance in zip(self.faces_position, self.mean_areas["object"]):
                data_dict[position].append(distance)

        return data_dict

    def debug(self):
        print("\n\n")
        print(f"Prediction Shape: {self.prediction.shape}")
        print(f"Midas Area : ")
        for idx, area in enumerate(self.midas_area):
            print(f"Faces {idx + 1} : {area.shape}")
        print()
        print(f"Faces : {self.faces}")


def main(yolo_type="yolov5s"):
    # Read the references image
    ref_image = cv2.imread('../ref_image.jpg')

    # Process face detector on reference image
    ref_fd = FaceDetector()
    ref_fd.set_frame(ref_image)
    ref_fd.get_face_information()

    # Process distance on reference image
    ref_dist = FaceDistanceEstimator(ref_fd)
    ref_dist.get_focal_length()

    # Define SurroundAudio object - face
    speaker_face = SurroundAudio()
    speaker_face.set_audio("../sound/person-ding.mp3")
    speaker_face.set_min_max(minimum=30, maximum=100)
    speaker_face.play_audio()
    # Define SurroundAudio object - object
    speaker_object = SurroundAudio()
    speaker_object.set_audio("../sound/object-ding.mp3")
    speaker_object.set_min_max(minimum=30, maximum=100)
    speaker_object.play_audio()

    # Define FaceDetector
    fd = FaceDetector()

    # Create object of FaceDistanceEstimator
    dist = FaceDistanceEstimator(fd)
    # Set the focal length value to object
    dist.focal_length = ref_dist.focal_length

    # Define ObjectDetector
    od = ObjectDetector()
    od.set_model(yolo_type)

    # Create object of FaceDistanceEstimator - Object
    dist_object = FaceDistanceEstimator(od)
    # Set the focal length value to object - Object
    dist_object.focal_length = ref_dist.focal_length

    fps = 1
    # Start the camera
    cap = cv2.VideoCapture(0)
    time_start = time.time()

    while True:
        ret, frame = cap.read()

        # Set frame of FaceDetector & Get information about the face
        fd.set_frame(frame)
        fd.get_face_information()

        # Calculating distance with FaceDistanceEstimator
        # Get distance of face
        dist.get_distance()

        # Detect objects
        od.set_frame(frame)
        od.get_object_information()
        dist_object.get_distance()

        # Create object of Image Controller
        im_controller = ImageController(f=frame)
        # Set the 'faces' inside the ImageController object
        im_controller.set_faces(fd.faces)
        # Set the 'distance' inside the ImageController object
        im_controller.set_distance(dist.distances)
        # Set the 'objects_box' inside the ImageController object
        im_controller.set_objects_box(od.boxes)
        # Set the 'objects_distance' inside the ImageController object
        im_controller.set_objects_distance(dist_object.distances)
        # Set the 'objects_categories' inside the ImageController object
        im_controller.set_objects_categories(od.categories)
        # Process & Show the frame
        im_controller.annotate(t="normal")
        im_controller.show(title="Normal Focal Length Estimation - press 'Q' to quit")

        # Get the distance value in form of dictionary
        distance_dict_faces = dist.get_face_distance_approx()

        # Get the distance value in form of dictionary - Object
        distance_dict_object = dist_object.get_face_distance_approx()

        # Compute the distance with 8D audio
        speaker_face.compute(distance_dict_faces)
        speaker_object.compute(distance_dict_object)

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
        # Reset the ObjectDetector attributes
        od.reset()
        # Reset the FaceDistanceEstimator - Object attributes
        dist_object.reset()

        # quit the program if you press 'q' on keyboard
        if cv2.waitKey(1) == ord("q"):
            speaker_face.stop_audio()
            speaker_object.stop_audio()
            break

    cap.release()
    cv2.destroyAllWindows()


def midas(model_type="midas_v21_small_256", yolo_type="yolov5s"):
    # Define SurroundAudio object - face
    speaker_face = SurroundAudio()
    speaker_face.set_audio("../sound/person-ding.mp3")
    speaker_face.set_min_max(minimum=30, maximum=100)
    speaker_face.play_audio()
    # Define SurroundAudio object - object
    speaker_object = SurroundAudio()
    speaker_object.set_audio("../sound/object-ding.mp3")
    speaker_object.set_min_max(minimum=30, maximum=100)
    speaker_object.play_audio()

    # Define FaceDetector
    fd = FaceDetector()
    # Define the Midas
    mi = Midas(model_type)
    # Define ObjectDetector
    od = ObjectDetector()
    od.set_model(yolo_type)

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

            # Midas
            # Setting faces coordinate
            mi.set_faces(fd.faces)
            mi.set_faces_position(fd.positions)

            # Detect objects
            od.set_frame(frame)
            od.get_object_information()
            # Set the object boxes after 'od' get the information about the objects
            mi.set_objects_boxes(od.boxes)
            # Set the object position
            mi.set_objects_position(od.positions)

            # Cut the bounding boxes area and calculate the mean of it
            mi.cut_midas_area()
            mi.calculate_mean_area()

            # Create object of Image Controller
            im_controller = ImageController(f=mi.prediction)
            # Set the 'faces' inside the ImageController object
            im_controller.set_faces(fd.faces)
            # Set the 'mean_area' inside the ImageController object
            im_controller.set_mean_area(mi.mean_areas["face"])
            # Set the 'objects_box' inside the ImageController object
            im_controller.set_objects_box(od.boxes)
            # Set the 'objects_distance' inside the ImageController object
            im_controller.set_objects_distance(mi.mean_areas["object"])
            # Set the 'objects_categories' inside the ImageController object
            im_controller.set_objects_categories(od.categories)
            # Process & Show the frame
            im_controller.annotate(t="depthmap")
            im_controller.show(title="Depth Map Estimation - press 'Q' to quit")

            # Get the distance value from depth map
            distance_dict_depth_face = mi.get_face_distance_approx(t="face")
            distance_dict_depth_object = mi.get_face_distance_approx(t="object")

            # Compute the distance with 8D audio - face
            speaker_face.compute(distance_dict_depth_face)
            # Compute the distance with 8D audio - object
            speaker_object.compute(distance_dict_depth_object)

            # FPS measurement
            alpha = 0.1
            if time.time() - time_start > 0:
                fps = (1 - alpha) * fps + alpha * 1 / (time.time() - time_start)  # exponential moving average
                time_start = time.time()
            print(f"\rFPS: {round(fps, 2)}", end="")

            # Reset the Midas attributes
            mi.reset()
            # Reset the FaceDetector attributes
            fd.reset()
            # Reset the ObjectDetector attributes
            od.reset()

            # quit the program if you press 'q' on keyboard
            if cv2.waitKey(1) == ord("q"):
                speaker_face.stop_audio()
                speaker_object.stop_audio()
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create a banner
    banner_text = pyfiglet.figlet_format("C-Eyes", font="slant")
    print(banner_text)
    print("Precision in Vision")
    print("FOR Compfest")
    print()

    # Get user input for menu selection
    while True:
        # Create a menu
        print("\nMenu:")
        print("1. Normal Focal Length Estimation")
        print("2. Depth Map Estimation (proper lighting required, no object other than face in front of camera)")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ")

        if choice == "1":
            print("You selected Normal Focal Length Estimation")
            available_model = ["5n", "5s", "5m", "5l", "5x"]
            description = ["poor", "ok", "medium", "good", "excellent, GPU recommended"]

            # Print the list of files
            print(f"Select {len(available_model)} YOLOv5 pre-trained model below:")
            for idx, file in enumerate(available_model):
                print(f"{idx + 1}. yolov{file} ({description[idx]})")

            choice = int(input("Enter your choice: "))
            main(f"yolov{available_model[choice - 1]}")
        elif choice == "2":
            print("You selected Depth Map Estimation")
            folder_path = 'weights/'
            extension = '.pt'
            # List all files in the folder with the specified extension
            files = [file[:-3] for file in os.listdir(folder_path) if file.endswith(extension)]
            # Print the list of files
            print(f"Select {len(files)} MiDAS pre-trained model below:")
            for idx, file in enumerate(files):
                print(f"{idx + 1}. {file}")

            choice = int(input("Enter your choice: "))
            midas(files[choice - 1])
        elif choice == "3":
            print("Exiting...")
            print("Thank You")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
