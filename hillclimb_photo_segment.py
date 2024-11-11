import cv2
import os
import numpy as np
import numpy.typing as npt
from typing import Annotated, List
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.engine.results import Results
from ultralytics.engine.model import Model
from midpoints import calculate_center_of_mass, find_midpoints_in_polygon, display_gripper
from node import Node, calculate_intersection_area, get_minimum_bounding_box
from yolo_utils.setup_yolo import setup_yolo
from raycasting import find_closest_intersection

# setup yolo model
model_name = 'yolo11n-seg.pt'
setup_yolo(model_name)
# Load the YOLO model
model: Model = YOLO(model_name)

# Load the image
image_path = 'pen.jpg'
image_path = os.path.abspath(image_path)
frame = cv2.imread(image_path)
if frame is None:
    print(f"Failed to load image: {image_path}")
    exit()
# names = model.names # This is never getting used

font = cv2.FONT_HERSHEY_SIMPLEX
# Process the image
results: List[Results] = model(frame)
result = results[0]
annotator = Annotator(frame, line_width=2)

if results and results[0].masks is not None and results[0].boxes is not None:
    clss = results[0].boxes.cls.tolist()
    masks = results[0].masks.xy
    edge_points: List[Annotated[npt.NDArray[np.int32], (2,)]] = [] # Moved to outside of loop to ensure edge_points is not Unbound
    
    # TODO: Turn this into a function to improve the readability
    for mask, cls in zip(masks, clss):
        color = colors(int(cls), True)
        txt_color = annotator.get_txt_color(color)

        # Convert the mask to integer points
        mask: npt.NDArray[np.int32] = np.array(mask, dtype=np.int32)

        # Calculate the total perimeter of the polygon
        total_perimeter: np.float64 = np.float64(0)
        for i in range(len(mask)):
            start_point: npt.NDArray[np.float64] = mask[i]
            end_point: npt.NDArray[np.float64] = mask[(i + 1) % len(mask)]  # Wrap around to the first point
            total_perimeter += np.linalg.norm(start_point - end_point)

        # Define the desired spacing between points
        SPACING: int = 5

        # Interpolate points at regular intervals along the perimeter
        current_distance: np.float64 = np.float64(0)
        for i in range(len(mask)):
            start_point: npt.NDArray[np.float64] = mask[i]
            end_point: npt.NDArray[np.float64] = mask[(i + 1) % len(mask)]  # Wrap around to the first point
            segment_length: np.float64 = np.linalg.norm(start_point - end_point)

            while current_distance + segment_length >= SPACING:
                if (segment_length == 0):
                    print("Error: segment_length is 0. Cannot divide by 0.")
                    break

                ratio: np.float64 = (SPACING - current_distance) / segment_length
                point_float: npt.NDArray[np.float64] = start_point + ratio * (end_point - start_point)

                # Cast point into a np.int32
                point: npt.NDArray[np.int32] = np.round(point_float).astype(np.int32)
                cv2.circle(frame, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 0), thickness=-1)
                edge_points.append(point)
                start_point = point
                segment_length -= SPACING - current_distance
                current_distance = np.float64(0)

            current_distance += segment_length
    
    center = calculate_center_of_mass(edge_points=edge_points)

    # Draw the center of mass
    cv2.circle(frame, tuple(center), 5, (255, 0, 0), -1)
    
    # Hill climb to best gripper pose by drawing a line from the center of mass to the polygon edge
    initial = find_closest_intersection(center=center, polygon_points=np.array(edge_points))
    magnitude = np.linalg.norm(initial - center)
    
    # normalize initial
    initial = initial / np.linalg.norm(initial)
    print("Initial Directions:", initial)

    # draw initial direction
    cv2.arrowedLine(frame, tuple(center), tuple(center + (initial * magnitude).astype(int)), (100, 100, 100), 2)
    # current = initial
    current = Node(direction=initial, center=center, edgepoints=edge_points)
    i = 0

    while True:
        i += 1
        print("runs:", i)
        # find gripper polygons in current direction
        # gripper_polygons = gripper_pose(current, center, edge_points)

        # # find num of midpoints in gripper (should correlate to higher value = more surface area to grab)   
        # current_value = find_midpoints_in_polygon(gripper_polygons[0], edge_points) + find_midpoints_in_polygon(gripper_polygons[1], edge_points)
        # step_angle = 10 * np.pi / 180  # 10 degrees
        # rotation_matrix_cw = np.array([[np.cos(step_angle), np.sin(step_angle)],
        #                                 [-np.sin(step_angle), np.cos(step_angle)]])
        # rotation_matrix_ccw = np.array([[np.cos(step_angle), -np.sin(step_angle)],
        #                                 [np.sin(step_angle), np.cos(step_angle)]])           
        
        # # find neighbors based on 10 degrees clockwise and counter clockwise
        # neighbor_cw = np.dot(rotation_matrix_cw, current)
        # neighbor_ccw = np.dot(rotation_matrix_ccw, current)

        # # normalize the direction
        # neighbor_cw = neighbor_cw / np.linalg.norm(neighbor_cw)
        # neighbor_ccw = neighbor_ccw / np.linalg.norm(neighbor_ccw)
        
        # print("Directions: \t cw:", neighbor_cw, "ccw:", neighbor_ccw)
        
        # # find neighboring gripper polygons
        # cw_polygons = gripper_pose(neighbor_cw, center, edge_points) 
        # ccw_polygons = gripper_pose(neighbor_ccw, center, edge_points)

        # # calculate midpoints in each gripper polygon of each neighbor
        # cw_value = find_midpoints_in_polygon(cw_polygons[0], edge_points) + find_midpoints_in_polygon(cw_polygons[1], edge_points)
        # ccw_value = find_midpoints_in_polygon(ccw_polygons[0], edge_points) + find_midpoints_in_polygon(ccw_polygons[1], edge_points)
        
        # # Maybe use the max height/width of the points in edge_points?
        # minimum_bounding_box = get_minimum_bounding_box(edge_points)

        # # Calculate intersection areas
        # cw_intersection_area = calculate_intersection_area(cw_polygons[0], minimum_bounding_box) + calculate_intersection_area(cw_polygons[1], minimum_bounding_box)
        # ccw_intersection_area = calculate_intersection_area(ccw_polygons[0], minimum_bounding_box) + calculate_intersection_area(ccw_polygons[1], minimum_bounding_box)
        
        # # Adjust values based on intersection areas
        # cw_value -= cw_intersection_area
        # ccw_value -= ccw_intersection_area
        
        # max_neighbor = max(cw_value, ccw_value)
        
        # if max_neighbor > current_value:
        #     if max_neighbor == cw_value:
        #         print("moving cw from current:", current_value, " to:", cw_value, "Where direction current:", current, "new direction: ", neighbor_cw)
        #         current = neighbor_cw
        #         continue
        #     if max_neighbor == ccw_value:
        #         print("moving ccw from current:", current_value, " to:", ccw_value, "Where direction current:", current, "new direction: ", neighbor_ccw)
        #         current = neighbor_ccw
        #         continue
        # elif max_neighbor <= current_value:
        #     print("found max at: ", max_neighbor)
        #     break
        new_node = current.compare_neighbor()
        if (new_node is current):
            break
        current = new_node
        
    current_value = current.value
    gripper_polygons = current.gripper_polygons
    cw_value = current.find_neighbor(10).value
    ccw_value = current.find_neighbor(-10).value
    current.display(frame)
    current = current.direction

            
    # Display the number of midpoints, gripper rectangles, and ending arrow
    print("Final direction: ", current)
    cv2.arrowedLine(frame, tuple(center), tuple(center + (current * magnitude).astype(int)), (255, 0, 0), 2)
    
    cv2.putText(frame, f'Midpoints in Rects: {current_value}', (10, 30), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Midpoints in cw: {cw_value}', (10, 50), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f'Midpoints in ccw: {ccw_value}', (10, 70), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

# Add information to quit to frame
cv2.putText(frame, text="Press any key to quit", org=(0, frame.shape[0] - 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))

# Display the annotated frame
cv2.imshow("YOLO Inference", frame)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed
cv2.destroyAllWindows()