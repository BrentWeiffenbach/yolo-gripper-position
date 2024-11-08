import os
from typing import List
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from raycasting import find_last_intersection
from midpoints import get_midpoints, find_midpoints_in_polygon, gripper_pose
from yolo_utils.setup_yolo import setup_yolo

def mouse_callback(event, x, y, flags, param):
    global mouse_click_coords
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_click_coords = np.array([x,y])
# setup yolo model
model_name = 'yolo11n-seg.pt'
setup_yolo(model_name)
# Load the YOLO model
model: Model = YOLO(model_name)

# capture video
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

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
        result = results[0]
        if result.masks:
            mask = result.masks[0]  # Get only the first found mask
            if mask.xy:
                segment = mask.xy[0] # get only the first segment
                # segement the mask to find midpoints
                steps = 2
                midpoints = np.array(segment, dtype=np.int32)
                for _ in range(steps):
                    midpoints = get_midpoints(midpoints)
                
                # draw the midpoints
                for midpoint in midpoints:
                    midpoint_int = tuple(map(int, midpoint))  # Convert to integer
                    cv2.circle(frame, center=midpoint_int, radius=4, color=(0, 255, 0), thickness=-1)

                # Draw the polygon outline
                cv2.polylines(frame, [np.array(midpoints, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

                # Calculate the moments of the polygon for center of mass
                moments = cv2.moments(np.array(midpoints, dtype=np.int32))

                # Compute the center of mass
                if moments["m00"] != 0:
                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])
                    center=np.array([cX, cY])
                else:
                    cX, cY = 0, 0
                # Draw the center of mass
                cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

                # Hill climb to best gripper pose to by drawing a line from the center of mass to the polygon edge
                initial = np.random.randint(0, frame.shape[1], size=2) # random initial direction
                current = initial
                while True:
                    # TODO probbaly make a class Node or Gripper_pose or something to make this cleaner
                    
                    # find gripper polygons in current direction
                    gripper_polygon = gripper_pose(current, center, midpoints)   
                    # find num of midpoints in gripper (should correlate to higher value = more surface area to grab)   
                    current_value = find_midpoints_in_polygon(gripper_polygon[0], midpoints) + find_midpoints_in_polygon(gripper_polygon[1], midpoints)
                    step_angle = np.pi / 18 # 10 degrees
                    rotation_matrix_cw = np.array([[np.cos(step_angle), np.sin(step_angle)],
                                                [-np.sin(step_angle), np.cos(step_angle)]])
                    rotation_matrix_ccw = np.array([[np.cos(step_angle), -np.sin(step_angle)],
                                                [np.sin(step_angle), np.cos(step_angle)]])           
                    
                    # find neighbors based on 10 degrees clockwise and counter clockwise
                    neighbor_cw = np.dot(rotation_matrix_cw, current)
                    neighbor_ccw = np.dot(rotation_matrix_ccw, current)
                    # normalize the direction
                    neighbor_cw = neighbor_cw / np.linalg.norm(neighbor_cw)
                    neighbor_ccw = neighbor_ccw / np.linalg.norm(neighbor_ccw)
                    
                    # find neihboring gripper polygons
                    cw_polygon = gripper_pose(neighbor_cw, center, midpoints) 
                    ccw_polygon = gripper_pose(neighbor_ccw, center, midpoints) 
                    # caluclate midpoints in each gripper polygon of each neighbor
                    cw_value = find_midpoints_in_polygon(cw_polygon[0], midpoints) + find_midpoints_in_polygon(cw_polygon[1], midpoints)
                    ccw_value = find_midpoints_in_polygon(ccw_polygon[0], midpoints) + find_midpoints_in_polygon(ccw_polygon[1], midpoints)
                    max_neighbor = max(cw_value, ccw_value)
                    
                    if max_neighbor == cw_value:
                        neighbor = neighbor_cw
                    if max_neighbor == ccw_value:
                        neighbor = neighbor_ccw
                    if max_neighbor <= current_value:
                        break
                    current = neighbor # if a neighbor is greater midpoints use neighbor as current
                        
                        
                # Display the number of midpoints within the rectangles
                # find gripper polygons in current direction
                gripper_polygon = gripper_pose(current, center, midpoints)   
                # find num of midpoints in gripper (should correlate to higher value = more surface area to grab)   
                current_value = find_midpoints_in_polygon(gripper_polygon[0], midpoints) + find_midpoints_in_polygon(gripper_polygon[1], midpoints)
                # cv2.line(frame, tuple(center), tuple(intersection_one.astype(int)), (255, 0, 0), 2)
                # cv2.line(frame, tuple(center), tuple(intersection_two.astype(int)), (255, 0, 0), 2)
                # Draw the rectangles
                cv2.polylines(frame, [gripper_polygon[0]], isClosed=True, color=(0, 255, 255), thickness=2)
                cv2.polylines(frame, [gripper_polygon[1]], isClosed=True, color=(0, 255, 255), thickness=2)
                cv2.putText(frame, f'Midpoints in Rects: {current_value}', (10, 30), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                            
        
        # Add information to quit to frame
        cv2.putText(frame, text="Press 'q' to quit", org=(0, frame.shape[0] - 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))

        # Display the annotated frame
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