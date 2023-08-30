import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a stereo object for stereo matching (e.g., Block Matching)
stereo = cv2.StereoBM_create(numDisparities=32, blockSize=41)

idx = 0
while True:
    if idx == 0:
        # Capture left and right frames (simulated by shifting)
        ret, left_frame = cap.read()
        idx += 1
    else:
        idx += 1

    print(f"idx : {idx}")
    if idx != 0 and idx % 500 == 0:
        print("executed")
        print(f"idx : {idx}")
        ret, right_frame = cap.read()

        # Convert frames to grayscale (necessary for stereo matching)
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Compute disparities using stereo matching
        disparities = stereo.compute(left_gray, right_gray)

        # Convert disparities to a depth map (adjust calibration parameters)
        depth_map = cv2.convertScaleAbs(disparities, alpha=0.5)

        # Display depth map
        cv2.imshow('Depth Map', depth_map)

        idx = 0

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
