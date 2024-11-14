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
image_path = 'scissors.jpg'
image_path = os.path.abspath(image_path)
frame = cv2.imread(image_path)
if frame is None:
    print(f"Failed to load image: {image_path}")
    exit()
frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
font = cv2.FONT_HERSHEY_SIMPLEX

# Variables to store the center and whether the center has been selected
center_selected = False
center = None

def mouse_callback(event, x, y, flags, param):
    global center_selected, center
    if event == cv2.EVENT_LBUTTONDOWN:
        center = np.array([x, y])
        center_selected = True

# Set mouse callback function
cv2.namedWindow("YOLO Inference")
cv2.setMouseCallback("YOLO Inference", mouse_callback)

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

    while True:
        if center is not None and center_selected:
            # Draw the center of mass
            cv2.circle(frame, tuple(center), 5, (255, 0, 0), -1)
            
            # Hill climb to best gripper pose by drawing a line from the center of mass to the polygon edge
            closest_point = find_closest_intersection(center=center, polygon_points=np.array(edge_points))
            magnitude = np.linalg.norm(closest_point - center)
            
            # normalize initial
            initial = (closest_point - center) / magnitude
            print("Initial Directions:", initial)

            # draw initial direction in gray
            cv2.arrowedLine(frame, tuple(center), tuple(center + (initial * magnitude).astype(int)), (100, 100, 100), 2)
            # current = initial
            current = Node(direction=initial, center=center, edgepoints=edge_points)
            i = 0
            # Hill climb to best value for gripper position
            prev_direction = None
            while True:
                i += 1
                print("runs:", i)
                new_node = current.compare_neighbor()
                if np.allclose(new_node.direction, current.direction, atol=1e-6) or (prev_direction is not None and np.allclose(new_node.direction, prev_direction, atol=1e-6)):
                    break
                prev_direction = current.direction
                current = new_node
                
            # conversions for print statements
            max_value = len(edge_points)
            print("max possible edge points", max_value)
            current_value = current.value
            gripper_polygons = current.gripper_polygons
            cw = current.find_neighbor(5)
            ccw = current.find_neighbor(-5)
            current.display(frame)
            
            # DEBUG PRINTS TO SEE NEIGHBORS AT THE END
            # cw.display(frame, color=(0, 0, 255))
            # ccw.display(frame, color=(0, 0, 255))
                    
            # Display the number of midpoints, gripper rectangles, and ending arrow
            print("Final direction: ", current.direction)
            cv2.arrowedLine(frame, tuple(center), tuple(center + (current.direction * magnitude).astype(int)), (255, 0, 0), 2)
            # display neighbor directions for debug
            # cv2.arrowedLine(frame, tuple(center), tuple(center + (cw.direction * magnitude).astype(int)), (255, 255, 255), 2)
            # cv2.arrowedLine(frame, tuple(center), tuple(center + (ccw.direction * magnitude).astype(int)), (255, 255, 255), 2)
            
            cv2.putText(frame, f'Midpoints in Rects: {current_value}', (10, 30), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Midpoints in cw: {cw.value}', (10, 50), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, f'Midpoints in ccw: {ccw.value}', (10, 70), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            
            # Reset center_selected to allow for new clicks
            center_selected = False

        # Add information to quit to frame
        cv2.putText(frame, text="Click to select center, press 'q' to quit", org=(0, frame.shape[0] - 10), fontFace=font, fontScale=0.5, color=(0, 0, 255))

        # Display the annotated frame
        cv2.imshow("YOLO Inference", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release resources
cv2.destroyAllWindows()