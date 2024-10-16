import os
import cv2

import numpy as np
import requests
from ultralytics import YOLO

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
model = YOLO(model_name)

# capture video
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        # annotated_frame = results[0].plot(boxes=False)
        for result in results:
            if result.masks:
                # for mask in result.masks:
                mask = result.masks[0]
                if mask.xy:
                    for segment in mask.xy:
                        segment = np.array(segment, dtype=np.int32)

                        # Draw line on the frame (webcam image)
                        cv2.polylines(frame, [segment], isClosed=False, color=(0, 255, 0), thickness=2)

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