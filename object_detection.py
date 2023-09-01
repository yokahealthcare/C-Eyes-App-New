import cv2

# Load YOLOv3 weights and configuration
yolo_net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO class names
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Create a blob from the frame and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward()

    # Loop over the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Get bounding box coordinates
                box = obj[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, w, h) = box.astype("int")

                # Draw bounding box and label
                color = (0, 255, 0)  # Green color
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the frame with detections
    cv2.imshow("YOLOv3 Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
