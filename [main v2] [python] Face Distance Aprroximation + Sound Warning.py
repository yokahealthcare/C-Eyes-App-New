# library
import cv2
import pygame
import numpy as np
from mtcnn.mtcnn import MTCNN

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

def get_width_pixel(image):

	face_width = [] # making face width to zero
	middle_x = []
	middle_y = []

	# converting color image to gray scale image
	
	# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# detecting face in the image
	# faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

	# detector is defined above, otherwise uncomment
	detector = MTCNN()
	# detect faces in the image
	faces = detector.detect_faces(image)

	for face in faces:
		print(f"Face : {face} \n\n")


	for face in faces:
		# looping through the faces detect in the image
		# getting coordinates x, y , width and height
		for (x, y, w, h) in [face['box']]:

			# draw the rectangle on the face
			cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)

			# getting face box middle coodinate
			middle_x.append(x+w/2)
			middle_y.append(y+h/2)

			# getting face width in the pixels
			face_width.append(w)

	# return the face width in pixel
	return face_width, list(zip(middle_x, middle_y))


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
	test_face_width_pixel, middle_x_y = get_width_pixel(frame)

	if test_face_width_pixel != 0:
		# get distance
		distance = get_distance(focal_length, known_width, test_face_width_pixel)

		# Drawing Text on the screen
		for idx,d in enumerate(distance):
			cv2.putText(frame, f"Person {idx+1}: {round(d,2)} CM", (30, 35+idx*30), fonts, 0.6, GREEN, 1)

	return frame, distance, middle_x_y



def url_to_image(url):
  image = io.imread(url)
  # return the image
  return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



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
	ref_face_width_pixel, _ = get_width_pixel(ref_image)
	# calculate the focal length
	# the index 1, because it detect two faces arrogantly, the correct one at index 1
	focal_length = get_focal_length(known_distance, known_width, ref_face_width_pixel[0])
	print(f"Focal Length has been Founded \t: {focal_length}\n")

	left_speaker = 0.0
	right_speaker = 0.0

	# Start the camera
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()

		frame_height = frame.shape[0]
		frame_width = frame.shape[1]

		frame, distance, middle_x_y = get_face_distance_approx(focal_length, known_width, frame)
		cv2.imshow("frame", frame)

		if distance != []:
			# normalize the distance range 0 to 1
			MIN = 30 	# centimeter
			MAX = 150 	# centimeter	 

			distance = (min(distance) - MIN) / (MAX - MIN)
			if distance < 1 and distance > 0:
				# added to volume in reverse, the bigger the distance the lower the volume
				volume = 1.0 - distance

			# # convert to numpy array
			# distance = np.array(distance)
			# min_index = np.argmin(distance)

			# distance = (distance[min_index] - MIN) / (MAX - MIN)
			# if distance < 1 and distance > 0:
			# 	# added to volume in reverse, the bigger the distance the lower the volume
			# 	if middle_x_y[min_index][0] < frame_width:
			# 		left_speaker = 1.0 - distance 
			# 	if middle_x_y[min_index][0] > frame_width:
			# 		right_speaker = 1.0 - distance 
				
		else:
			if volume > 0.0:
				# slowly decrease volume if no face detected
				volume -= 0.003

				
		# set volume
		pygame.mixer.music.set_volume(volume)
		print(f"Current Volume : {pygame.mixer.music.get_volume()}")

		# quit the program if you press 'q' on keyboard
		if cv2.waitKey(1) == ord("q"):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()