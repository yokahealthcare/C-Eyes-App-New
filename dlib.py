import dlib
import cv2

# Load the pre-trained face detection model from dlib
detector = dlib.get_frontal_face_detector()

# Read an image or capture a frame from a camera
image = cv2.imread('image.jpg')

# Detect faces in the image
faces = detector(image)

# Draw rectangles around the detected faces
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the image with detected faces
cv2.imshow('Face Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
