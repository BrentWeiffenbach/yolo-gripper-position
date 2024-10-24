import os
from typing import List, Literal
import cv2

import numpy as np
import requests
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from raycasting import find_last_intersection

#  Set up the yolo model
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

camera_source: Literal[0] | Literal[1] = 0
"""The camera source to use. Integer of either 0 or 1
"""

# capture video
cap = cv2.VideoCapture(camera_source)
font = cv2.FONT_HERSHEY_SIMPLEX

# Variables to store mouse click coordinates
mouse_click_coords = None

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

def mouse_callback(event, x, y, flags, param):
    global mouse_click_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_coords = np.array([x,y])

# Set mouse callback function
cv2.namedWindow("YOLO Inference")
cv2.setMouseCallback("YOLO Inference", mouse_callback)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO inference on the frame
        results: List[Results] = model(frame)

        # Visualize the results on the frame
        for result in results:
            if result.masks:
                mask = result.masks[0]  # Get only the first found mask
                if mask.xy:
                    for segment in mask.xy:
                        steps = 2
                        midpoints = np.array(segment, dtype=np.int32)
                        for _ in range(steps):
                            midpoints = get_midpoints(midpoints)
                        
                        for midpoint in midpoints:
                            midpoint_int = tuple(map(int, midpoint))  # Convert to integer
                            cv2.circle(frame, center=midpoint_int, radius=4, color=(0, 255, 0), thickness=-1)

                        # Draw the polygon outline
                        cv2.polylines(frame, [np.array(midpoints, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

                        # Calculate the moments of the polygon
                        M = cv2.moments(np.array(midpoints, dtype=np.int32))

                        # Compute the center of mass
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            center=np.array([cX, cY])
                        else:
                            cX, cY = 0, 0
                        # Draw the center of mass
                        cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

                        # Draw a line from the center of mass to the polygon edge in direction of mouse click
                        if mouse_click_coords is not None:
                            reversed_mouse_click_coords = 2 * center - mouse_click_coords
                            intersection_one = find_last_intersection(center=center, direction_pos=mouse_click_coords, polygon_points=np.array(midpoints))
                            intersection_two = find_last_intersection(center=center, direction_pos=reversed_mouse_click_coords, polygon_points=np.array(midpoints))
                            cv2.line(frame, tuple(center), tuple(intersection_one.astype(int)), (255, 0, 0), 2)
                            cv2.line(frame, tuple(center), tuple(intersection_two.astype(int)), (255, 0, 0), 2)
                            
                            # Calculate the direction vector of the intersection line
                            direction_vector_one = intersection_one - center
                            direction_vector_two = intersection_two - center
                            direction_vector_one = direction_vector_one / np.linalg.norm(direction_vector_one)  # Normalize the vector
                            direction_vector_two = direction_vector_two / np.linalg.norm(direction_vector_two)  # Normalize the vector

                            # Calculate the perpendicular vector
                            perpendicular_vector_one = np.array([-direction_vector_one[1], direction_vector_one[0]])
                            perpendicular_vector_two = np.array([-direction_vector_two[1], direction_vector_two[0]])

                            # Define the rectangle size
                            rect_width = 60
                            rect_height = 10

                            # Calculate the four corners of the rectangle on one vector 
                            rect_points_one = np.array([
                                intersection_one + rect_width / 2 * perpendicular_vector_one + rect_height / 2 * direction_vector_one,
                                intersection_one - rect_width / 2 * perpendicular_vector_one + rect_height / 2 * direction_vector_one,
                                intersection_one - rect_width / 2 * perpendicular_vector_one - rect_height / 2 * direction_vector_one,
                                intersection_one + rect_width / 2 * perpendicular_vector_one - rect_height / 2 * direction_vector_one
                            ], dtype=np.int32)
                            
                            # Calculate the four corners of the rectangle on two vector
                            rect_points_two = np.array([
                                intersection_two + rect_width / 2 * perpendicular_vector_two + rect_height / 2 * direction_vector_two,
                                intersection_two - rect_width / 2 * perpendicular_vector_two + rect_height / 2 * direction_vector_two,
                                intersection_two - rect_width / 2 * perpendicular_vector_two - rect_height / 2 * direction_vector_two,
                                intersection_two + rect_width / 2 * perpendicular_vector_two - rect_height / 2 * direction_vector_two
                            ], dtype=np.int32)

                            # Draw the rectangles
                            cv2.polylines(frame, [rect_points_one], isClosed=True, color=(0, 255, 255), thickness=2)
                            cv2.polylines(frame, [rect_points_two], isClosed=True, color=(0, 255, 255), thickness=2)
                            
                            # Number of midpoints within the rectangle
                            # Check if midpoints are within the rectangles
                            midpoints_within_rect_one = sum(cv2.pointPolygonTest(rect_points_one, (float(midpoint[0]), float(midpoint[1])), False) >= 0 for midpoint in midpoints)
                            midpoints_within_rect_two = sum(cv2.pointPolygonTest(rect_points_two, (float(midpoint[0]), float(midpoint[1])), False) >= 0 for midpoint in midpoints)

                            # Display the number of midpoints within the rectangles
                            cv2.putText(frame, f'Midpoints in Rect 1: {midpoints_within_rect_one}', (10, 30), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                            cv2.putText(frame, f'Midpoints in Rect 2: {midpoints_within_rect_two}', (10, 50), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                            

        # Add information to quit to frame
        cv2.putText(frame, text="Press 'q' to quit", org=(0, frame.shape[0] - 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))
        cv2.putText(frame, text="Press 'c' to change camera source", org=(0, frame.shape[0] - 30), fontFace=font, fontScale=0.5, color=(0, 0, 255))

        # Display the annotated frame
        cv2.imshow("YOLO Inference", frame)

        # Check to see if there is a pressed key
        key = cv2.waitKey(1) & 0xFF

        # Break the loop if 'q' is pressed
        if key == ord("q"):
            break
        
        # Switch camera if 'c' is pressed
        if key & 0xFF == ord("c"):
            camera_source = 1 - camera_source # type: ignore
            cap = cv2.VideoCapture(camera_source)
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()