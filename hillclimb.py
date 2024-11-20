import cv2
import os
import numpy as np
import numpy.typing as npt
from typing import Annotated, List
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.engine.results import Results
from ultralytics.engine.model import Model
from midpoints import calculate_center_of_mass
from node import Node
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
annotator = Annotator(frame, line_width=2)

def get_edge_points(frame: cv2.typing.MatLike, annotator: Annotator, clss: List, masks: npt.NDArray[np.float32]) -> List[Annotated[npt.NDArray[np.int32], (2,)]]:
    """Gets the edge points of a given frame

    Returns:
        List[NDArray[int32]]: The edge points of the masks
    """
    edge_points: List[Annotated[npt.NDArray[np.int32], (2,)]] = []
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
    return edge_points


def hill_climb(results: List[Results], frame: cv2.typing.MatLike):
    if not results or results[0].masks is None or results[0].boxes is None:
        return
    
    clss = results[0].boxes.cls.tolist()
    masks = np.array(results[0].masks[0].xy) # This is actually a NDArray[NDArray[NDArray[float32]]]

    edge_points: List[Annotated[npt.NDArray[np.int32], (2,)]] = get_edge_points(frame, annotator, clss, masks)
    
    center = calculate_center_of_mass(edge_points=edge_points)

    # Draw the center of mass
    cv2.circle(frame, tuple(center), 5, (255, 0, 0), -1)
    # Hill climb to best gripper pose by drawing a line from the center of mass to the polygon edge
    closest_point = find_closest_intersection(center=center, polygon_points=np.array(edge_points))
    magnitude = np.linalg.norm(closest_point - center)
    
    # normalize initial
    initial = (closest_point - center) / magnitude
    # print("Initial Directions:", initial) #DEBUG

    # draw initial direction in gray
    cv2.arrowedLine(frame, tuple(center), tuple(center + (initial * magnitude).astype(int)), (100, 100, 100), 2)
    # current = initial
    current = Node(direction=initial, center=center, edgepoints=edge_points)
    # print("intial: ", current.calculate_value())
    i = 0
    # Hill climb to best value for gripper position
    prev_direction = None
    while True:
        i += 1
        # print("runs:", i) # DEBUG
        new_node = current.compare_neighbor()
        if np.allclose(new_node.direction, current.direction, atol=1e-6): # or (prev_direction is not None and np.allclose(new_node.direction, prev_direction, atol=1e-6)):
            break
        # prev_direction = current.direction
        current = new_node
        
    # conversions for print statements
    max_value = len(edge_points)
    # print("max possible edge points", max_value) #DEBUG
    current_value = current.value
    gripper_polygons = current.gripper_polygons
    cw = current.find_neighbor(10)
    ccw = current.find_neighbor(-10)
    current.display(frame)
    
    # DEBUG PRINTS TO SEE NEIGHBORS AT THE END
    # cw.display(frame, color=(0, 0, 255))
    # ccw.display(frame, color=(0, 0, 255))
            
    # Display the number of midpoints, gripper rectangles, and ending arrow
    # print("Final direction: ", current.direction) #DEBUG
    cv2.arrowedLine(frame, tuple(center), tuple(center + (current.direction * magnitude).astype(int)), (255, 0, 0), 2)
    # display neighbor directions for debug
    # cv2.arrowedLine(frame, tuple(center), tuple(center + (cw.direction * magnitude).astype(int)), (255, 255, 255), 2)
    # cv2.arrowedLine(frame, tuple(center), tuple(center + (ccw.direction * magnitude).astype(int)), (255, 255, 255), 2)
    
    cv2.putText(frame, f'Midpoints in Rects: {current_value}', (10, 30), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # DEBUGS:
    # cv2.putText(frame, f'Midpoints in cw: {cw.value}', (10, 50), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # cv2.putText(frame, f'Midpoints in ccw: {ccw.value}', (10, 70), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)