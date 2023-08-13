from facenet_pytorch import MTCNN
from PIL import Image
import torch
import cv2
import numpy as np

# colors
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device being used : {device}")

# Create face detector
mtcnn = MTCNN(keep_all=True, device=device)

cap = cv2.VideoCapture("video.mp4")

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

while cap.isOpened():

    success, ori_frame = cap.read()
    if not success:
        print("Video has ended!")
        break

    frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)

    # Detect face
    boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)

    # Convert the OpenCV image to a torch.Tensor
    # Convert the image from BGR to RGB format
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to torch.Tensor with shape (C, H, W)
    # frame_tensor = torch.from_numpy(frame_rgb.transpose((2, 0, 1))).float()

    # print(f"frame : {frame}")

    # boxes, probs, landmarks = mtcnn.detect(frame_tensor, landmarks=True)

    # faces = []

    # box_ind = int(i / self.stride)
    # if boxes[box_ind] is None:
    #     continue
    # for box in boxes[box_ind]:
    #     box = [int(b) for b in box]

    print(f"Boxes : {boxes}")

    if boxes is not None:
        for box in boxes:
            #convert to int
            box = [int(x) for x in box]
            x_l, y_l, x_r, y_r = box
            cv2.rectangle(ori_frame, (x_l, y_l), (x_r, y_r), GREEN, 2)

    cv2.imshow("frame", ori_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 

# class FastMTCNN(object):
#     """Fast MTCNN implementation."""
    
#     def __init__(self, stride, resize=1, *args, **kwargs):
#         """Constructor for FastMTCNN class.
        
#         Arguments:
#             stride (int): The detection stride. Faces will be detected every `stride` frames
#                 and remembered for `stride-1` frames.
        
#         Keyword arguments:
#             resize (float): Fractional frame scaling. [default: {1}]
#             *args: Arguments to pass to the MTCNN constructor. See help(MTCNN).
#             **kwargs: Keyword arguments to pass to the MTCNN constructor. See help(MTCNN).
#         """
#         self.stride = stride
#         self.resize = resize
#         self.mtcnn = MTCNN(*args, **kwargs)
        
#     def __call__(self, frames):
#         """Detect faces in frames using strided MTCNN."""
#         if self.resize != 1:
#             frames = [
#                 cv2.resize(f, (int(f.shape[1] * self.resize), int(f.shape[0] * self.resize)))
#                     for f in frames
#             ]
                      
#         boxes, probs = self.mtcnn.detect(frames[::self.stride])

#         faces = []
#         for i, frame in enumerate(frames):
#             box_ind = int(i / self.stride)
#             if boxes[box_ind] is None:
#                 continue
#             for box in boxes[box_ind]:
#                 box = [int(b) for b in box]
#                 faces.append(frame[box[1]:box[3], box[0]:box[2]])
        
#         return faces


# fast_mtcnn = FastMTCNN(
#     stride=4,
#     resize=1,
#     margin=14,
#     factor=0.6,
#     keep_all=True,
#     device=device
# )


# def run_detection(fast_mtcnn, filenames):
#     frames = []
#     frames_processed = 0
#     faces_detected = 0
#     batch_size = 60
#     start = time.time()

#     v_cap = FileVideoStream(filenames).start()
#     v_len = int(v_cap.stream.get(cv2.CAP_PROP_FRAME_COUNT))

#     for j in range(v_len):

#         frame = v_cap.read()
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frames.append(frame)

#         if len(frames) >= batch_size or j == v_len - 1:

#             faces = fast_mtcnn(frames)

#             frames_processed += len(frames)
#             faces_detected += len(faces)
#             frames = []

#             print(
#                 f'Frames per second: {frames_processed / (time.time() - start):.3f},',
#                 f'faces detected: {faces_detected}\r',
#                 end=''
#             )

#     v_cap.stop()

# print("Starting...")
# run_detection(fast_mtcnn, filenames)