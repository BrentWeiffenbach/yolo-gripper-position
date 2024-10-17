import os
from typing import List
import cv2

import numpy as np
import requests
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results

model_name = 'yolo11n-seg.pt'
# Check if model exists
if not os.path.isfile(model_name):
    print(f'{model_name} does not exist. Downloading...')
    download_url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt'
    response = requests.get(download_url)

    if response.status_code == 200:
        with open(model_name, 'wb') as file:
            file.write(response.content)
        print(f'Downloaded {model_name}')
    else:
        print(f'Failed to download {model_name}')

# Load the YOLO model
model: Model = YOLO(model_name)

# capture video
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

def get_midpoints(coordinates: np.ndarray | list[tuple]) -> list[tuple[np.int32, np.int32]]:
    """This function takes a list of coordinates and creates a line between them. It returns the coordinates of the midpoint of said line.

    Args:
        coordinates (np.ndarray | list[tuple]): A list of coordinates to get the midpoints from.

    Returns:
        list[tuple[np.int32, np.int32]]: A list of coordinates containing the midpoints of their connecting lines.
    """
    midpoints: list[tuple[np.int32, np.int32]] = []
    
    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        
        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
        midpoints.append(midpoint)
    
    return midpoints[::2] # Halfs the number of midpoints by taking every other one

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results: List[Results] = model(frame)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot(boxes=False)
        for result in results:
            if result.masks:
                # for mask in result.masks:
                mask = result.masks[0] # Get only the first found mask
                if mask.xy:
                    for segment in mask.xy:
                        steps = 4
                        midpoints = np.array(segment, dtype=np.int32)
                        for _ in range(steps):
                            midpoints = get_midpoints(midpoints)
                        
                        for midpoint in midpoints:
                            midpoint_int = tuple(map(int, midpoint))  # Convert to integer

                            cv2.circle(frame, center=midpoint_int, radius=4, color=(0, 255, 0), thickness=-1)

                            # Use the following below to draw a smooth line of the outline
                            # cv2.polylines(frame, [segment], isClosed=False, color=(0, 255, 0), thickness=2)

        # Add information to quit to frame
        # cv2.putText(annotated_frame, text="Press 'q' to quit", org=(0, frame.shape[0] - 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))
        cv2.putText(frame, text="Press 'q' to quit", org=(0, frame.shape[0] - 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))

        # Display the annotated frame
        # cv2.imshow("YOLO Inference", annotated_frame)
        cv2.imshow("YOLO Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()