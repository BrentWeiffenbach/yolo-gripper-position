import cv2
import os
from pyinstrument import Profiler
from ultralytics import YOLO
from ultralytics.engine.model import Model
from yolo_utils.setup_yolo import setup_yolo
from typing import Literal
from hillclimb import hill_climb

camera_source: Literal[0] | Literal[1] = 0
model_name = 'yolo11s-seg.pt'
setup_yolo(model_name)
# Load the YOLO model
model: Model = YOLO(model_name)
font = cv2.FONT_HERSHEY_SIMPLEX
# person, frisbee, sports ball, bottle, cup, fork, knife, spoon, banana, apple, orange, carrot, mouse, remote, cell phone, book, vase, scissors, toothbrush
TRACKED_CLASSES = [0, 29, 39, 41, 42, 43, 44, 46, 47, 49, 51, 64, 65, 67, 73, 75, 76, 79]

# Ask user if it should use a video file or webcam
print("Would you like to grab objects on webcam, video, or a photo file?")
while True:
    setup_type = input("Enter 'webcam', 'video', or 'photo': ").strip().lower()
    if setup_type in ['webcam', 'video', 'photo']:
        break
    if not setup_type:
        setup_type = 'webcam' # Default to webcam
        break
    else:
        print("Invalid input. Please enter 'webcam', 'video', or 'photo'")

def find_file(directory: str, file_name: str) -> str:
    for root, _, files in os.walk(directory):
        if file_name in files:
            return os.path.join(root, file_name)
    return ""

# if setup is video, ask for video path
if setup_type == 'video' or setup_type == 'photo':
    # Path to the downloaded video
    file_name = input("Enter the name of the file to grab objects on (default is example.jpg/.mp4): ").strip()
    
    if not file_name:
        print("no file name provided, going with default")
        if setup_type == 'video':
            file_name = 'example.mp4'
        if setup_type == 'photo':
            file_name = 'example.jpg'
    print("looking for file:", file_name)
    file_path = find_file(os.getcwd(), file_name)
    
    if not os.path.isfile(file_path):
        print(f"Absolute path: {file_path}")
        print(f"Could not find file: {file_path}")
        exit(1)
    
    if file_path and setup_type == 'video':
        file_name = file_path
        print(f"Absolute path: {file_name}")
        # get video from path
        cap = cv2.VideoCapture(file_path)

    if file_path and setup_type == 'photo':
        # read photo from file
        frame = cv2.imread(file_path)
        # results: List[Results] = model(frame)
        results = model.track(frame, classes=TRACKED_CLASSES)
        hill_climb(results, frame)
        # Add information to quit to frame
        cv2.putText(frame, text="Press any key to quit", org=(0, frame.shape[0] - 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))
        
        # display class info in bottom right
        detected_classes = []
        if results[0].boxes is not None:
            detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
        class_text = ', '.join(detected_classes)
        cv2.putText(frame, text=f"Detected: {class_text}", org=(frame.shape[0], frame.shape[0] - 10), fontFace=font, fontScale=0.5, color=(0, 0, 255), thickness=1)

        # Display the annotated frame
        cv2.imshow("YOLO Inference", frame)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()
        exit(1)

# if setup type is webcam capture a picture
if setup_type == 'webcam':
    # capture picture
    cap = cv2.VideoCapture(camera_source)

profiler = Profiler()
profiler.start()
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        # frame = cv2.resize(frame, (640, 640))
        frame = frame[:-100, :]  # Crop the bottom 30 pixels
        # Run YOLO inference on the frame
        # results: List[Results] = model(frame)
        results = model.track(frame, classes=TRACKED_CLASSES)
        hill_climb(results, frame)
        cv2.putText(frame, text="Press q to quit", org=(0, frame.shape[0] - 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))
        if setup_type == 'webcam':
            cv2.putText(frame, text="Press c to switch webcam source", org=(0, frame.shape[0] - 30), fontFace=font, fontScale=0.5, color=(0, 0, 255))
            
        # display class info in bottom right
        detected_classes = []
        if results[0].boxes is not None:
            detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
        class_text = ', '.join(detected_classes)
        cv2.putText(frame, text=f"Detected: {class_text}", org=(0, frame.shape[0] - 50), fontFace=font, fontScale=0.5, color=(0, 0, 255), thickness=1)
    
    cv2.imshow("Gripper", frame)

    # Check to see if there is a pressed key
    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if key == ord("q"):
        break # TODO: Check if it is a video before showing this

    # Switch camera if 'c' is pressed
    if key & 0xFF == ord("c"):
        camera_source = 1 - camera_source # type: ignore
        cap = cv2.VideoCapture(camera_source)

profiler.stop()
profiler.print()
cv2.destroyAllWindows()