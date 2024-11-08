import os
from typing import List
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from midpoints import get_midpoints, find_midpoints_in_polygon, gripper_pose, display_gripper
from yolo_utils.setup_yolo import setup_yolo
from raycasting import find_closest_intersection

# setup yolo model
model_name = 'yolo11n-seg.pt'
setup_yolo(model_name)
# Load the YOLO model
model: Model = YOLO(model_name)

# Load the image
image_path = 'bottle.jpg'
image_path = os.path.abspath(image_path)
frame = cv2.imread(image_path)
if frame is None:
    print(f"Failed to load image: {image_path}")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX
# Process the image
results: List[Results] = model(frame)
result = results[0]
if result.masks:
    mask = result.masks[0]  # Get only the first found mask
    if mask.xy:
        segment = mask.xy[0]  # get only the first segment
        # segment the mask to find midpoints
        steps = 1
        midpoints = np.array(segment, dtype=np.int32)
        for _ in range(steps):
            midpoints = get_midpoints(midpoints)
        
        # draw the midpoints
        for midpoint in midpoints:
            midpoint_int = tuple(map(int, midpoint))  # Convert to integer
            cv2.circle(frame, center=midpoint_int, radius=4, color=(0, 255, 0), thickness=-1)

        # Draw the polygon outline
        # cv2.polylines(frame, [np.array(midpoints, dtype=np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        # Calculate the moments of the polygon for center of mass
        moments = cv2.moments(np.array(midpoints, dtype=np.int32))

        # Compute the center of mass
        if moments["m00"] != 0:
            cX = int(moments["m10"] / moments["m00"])
            cY = int(moments["m01"] / moments["m00"])
            center = np.array([cX, cY])
        else:
            cX, cY = 0, 0
        # Draw the center of mass
        cv2.circle(frame, (cX, cY), 5, (255, 0, 0), -1)

        # Hill climb to best gripper pose by drawing a line from the center of mass to the polygon edge
        # initial = np.random.randint(-3, frame.shape[1], size=2)  # random initial direction
        # initial = np.array([0, -1])
        initial = find_closest_intersection(center=center, polygon_points=np.array(midpoints))
        magnitude = np.linalg.norm(initial - center)
        # normalize
        initial = initial / np.linalg.norm(initial)
        print("Initial Directions:", initial)
        # draw initial direction
        cv2.arrowedLine(frame, tuple(center), tuple(center + (initial * magnitude).astype(int)), (100, 100, 100), 2)
        current = initial
        i=0
        while True:
            # TODO probably make a class Node or Gripper_pose or something to make this cleaner
            i+=1
            print("runs:", i)
            # find gripper polygons in current direction
            gripper_polygons = gripper_pose(current, center, midpoints)   
            # find num of midpoints in gripper (should correlate to higher value = more surface area to grab)   
            current_value = find_midpoints_in_polygon(gripper_polygons[0], midpoints) + find_midpoints_in_polygon(gripper_polygons[1], midpoints)
            step_angle = 10 * np.pi / 180  # 10 degrees
            rotation_matrix_cw = np.array([[np.cos(step_angle), np.sin(step_angle)],
                                           [-np.sin(step_angle), np.cos(step_angle)]])
            rotation_matrix_ccw = np.array([[np.cos(step_angle), -np.sin(step_angle)],
                                            [np.sin(step_angle), np.cos(step_angle)]])           
            
            # find neighbors based on 1 degrees clockwise and counter clockwise
            neighbor_cw = np.dot(rotation_matrix_cw, current)
            neighbor_ccw = np.dot(rotation_matrix_ccw, current)
            # normalize the direction
            neighbor_cw = neighbor_cw / np.linalg.norm(neighbor_cw)
            neighbor_ccw = neighbor_ccw / np.linalg.norm(neighbor_ccw)
            
            # draw direction for debuging
            # cv2.arrowedLine(frame, tuple(center), tuple(center + (neighbor_cw * 50).astype(int)), (255, 0, 0), 2)
            # cv2.arrowedLine(frame, tuple(center), tuple(center + (neighbor_ccw * 50).astype(int)), (255, 0, 0), 2)
            print("Directions: \t cw:", neighbor_cw, "ccw:", neighbor_ccw)
            
            # find neighboring gripper polygons
            cw_polygons = gripper_pose(neighbor_cw, center, midpoints) 
            ccw_polygons = gripper_pose(neighbor_ccw, center, midpoints) 
            # calculate midpoints in each gripper polygon of each neighbor
            cw_value = find_midpoints_in_polygon(cw_polygons[0], midpoints) + find_midpoints_in_polygon(cw_polygons[1], midpoints)
            ccw_value = find_midpoints_in_polygon(ccw_polygons[0], midpoints) + find_midpoints_in_polygon(ccw_polygons[1], midpoints)
            max_neighbor = max(cw_value, ccw_value)
            
            # debug drawings:
            # print("drawing debug")
            # Draw the rectangles
            # display_gripper(gripper_polygons, frame)
            # draw neighbors
            # display_gripper(cw_polygons, frame)
            # display_gripper(ccw_polygons, frame)
            
            if max_neighbor > current_value:
                if max_neighbor == cw_value:
                    print("moving cw from current:", current_value, " to:", cw_value, "Where direction current:", current, "new direction: ", neighbor_cw)
                    current = neighbor_cw
                    continue
                if max_neighbor == ccw_value:
                    print("moving ccw from current:", current_value, " to:", ccw_value, "Where direction current:", current, "new direction: ", neighbor_ccw)
                    current = neighbor_ccw
                    continue
            elif max_neighbor <= current_value:
                print("found max at: ", max_neighbor)
                break
                
        # Display the number of midpoints, gripper rectangles, and ending arrow
        print("Final direction: ", current)
        cv2.arrowedLine(frame, tuple(center), tuple(center + (current * magnitude).astype(int)), (255, 0, 0), 2)
        display_gripper(gripper_polygons, frame)
        cv2.putText(frame, f'Midpoints in Rects: {current_value}', (10, 30), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Midpoints in cw: {cw_value}', (10, 50), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Midpoints in ccw: {ccw_value}', (10, 70), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# Add information to quit to frame
cv2.putText(frame, text="Press any key to quit", org=(0, frame.shape[0] - 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))

# Display the annotated frame
cv2.imshow("YOLO Inference", frame)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()