# library
import cv2
import pygame
import numpy as np

# face detector object
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
	face_width = [] # making face width to zero

	for x,y,w,h in faces:
		# getting face width in the pixels
		face_width.append(w)

	# return the face width in pixel
	return face_width

def get_face_position(faces):

	position = []
	for x,y,w,h in faces:
		mx = x + int(w / 2)
		print(f"mx : {mx}")
		print(f"WIDTH : {WIDTH}")

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
		print(f"Face Position : {face_position}")
		# get distance
		distance = get_distance(focal_length, known_width, test_face_width_pixel)	

	data_list = zip(face_position, distance)

	data_dict = {"L": [], "R": []}
	for position, distance in data_list:
		data_dict[position].append(distance)

	return data_dict

def main():
	# initialize pygame
	pygame.init()
	pygame.mixer.init()

	# Get a free channel
	channel = pygame.mixer.find_channel()

	# Set the volume for left and right speakers
	left_volume = 0.0
	right_volume = 0.0
	# Set the volume for left and right speakers
	channel.set_volume(left_volume, right_volume)

	# Load and play the audio file
	sound = pygame.mixer.Sound('sound/object-ding.mp3')
	channel.play(sound, -1)

	# read the references image
	ref_image = cv2.imread('ref_image.jpg')
	# detect faces on references image
	faces = detect_faces(ref_image)
	print(f"Faces : {faces}")
	# getting the pixel width of the reference image first
	ref_face_width_pixel = get_width_pixel(faces)
	print(f"Ref Face Width Pixel : {ref_face_width_pixel}")

	# calculate the focal length
	# the index 1, because it detect two faces arrogantly, the correct one at index 1
	focal_length = get_focal_length(known_distance, known_width, ref_face_width_pixel[1])
	print(f"Focal Length has been Founded \t: {focal_length}\n")
	input("kkk")

	# Start the camera
	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()
	# update the WIDTH value of screen
	global WIDTH
	WIDTH = frame.shape[1]

	while True:
		ret, frame = cap.read()

		faces = detect_faces(frame)
		print(f"Faces : {faces}")
		distance = get_face_distance_approx(focal_length, known_width, faces)

		# Drawing Text on the screen
		#for idx,d in enumerate(distance):
			#cv2.putText(frame, f"Person {idx+1}: {round(d,2)} CM", (30, 35+idx*30), fonts, 0.6, GREEN, 1)
		
		cv2.imshow("Result of Detection", frame)

		# normalize the distance range 0 to 1
		MIN = 30 	# centimeter
		MAX = 100 	# centimeter

		print(f"Distance : {distance}")
		print(f"WIDTH : {WIDTH}")

		if distance["L"] != []:
				left = (min(distance["L"]) - MIN) / (MAX - MIN)
				left_volume = 1.0 - left

		else:
			if left_volume > 0.0:
				# slowly decrease volume if no face detected
				left_volume -= 0.05


		if distance["R"] != []:
				right = (min(distance["R"]) - MIN) / (MAX - MIN)
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
		# print(f"After - Left Volume : {left_volume}")
		# print(f"After - Right Volume : {right_volume}\n\n")

		# quit the program if you press 'q' on keyboard
		if cv2.waitKey(1) == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()