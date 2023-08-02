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

def get_width_pixel(image):

    face_width = [] # making face width to zero

    # converting color image to gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detecting face in the image
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    # looping through the faces detect in the image
    # getting coordinates x, y , width and height
    for (x, y, h, w) in faces:

        # draw the rectangle on the face
        cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)

        # getting face width in the pixels
        face_width.append(w)

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
	distance = None
	# get the width of pixel from the image
	test_face_width_pixel = get_width_pixel(frame)

	if test_face_width_pixel != 0:
		# get distance
		distance = get_distance(focal_length, known_width, test_face_width_pixel)

		# Drawing Text on the screen
		for idx,d in enumerate(distance):
			cv2.putText(frame, f"Person {idx+1}: {round(d,2)} CM", (30, 35+idx*30), fonts, 0.6, GREEN, 1)

	return frame, distance



def url_to_image(url):
  image = io.imread(url)
  # return the image
  return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)





ref_image = cv2.imread('ref_image.jpg')
# getting focal length of the reference image first
ref_face_width_pixel = get_width_pixel(ref_image)

# calculate the focal length
focal_length = get_focal_length(known_distance, known_width, ref_face_width_pixel[1])

print(f"Focal Length has been Founded \t: {focal_length}\n")


# Start the camera
cap = cv2.VideoCapture(0)
while True:
	ret, frame = cap.read()

	frame, distance = get_face_distance_approx(focal_length, known_width, frame)
	cv2.imshow("frame", frame)


	# quit the program if you press 'q' on keyboard
	if cv2.waitKey(1) == ord("q"):
		break