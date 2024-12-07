import cv2
import numpy as np
from numpy import int32, float32
from numpy.typing import NDArray
from typing import Annotated, Final
from ultralytics.utils.plotting import colors
from ultralytics.engine.results import Results
from midpoints import (calculate_center_of_mass, calculate_visual_center_of_polygon)
from node import Node
from raycasting import find_closest_intersection
import random
from shapely.geometry import Point, Polygon

def get_random_point_in_polygon(polygon_points: NDArray[int32]) -> NDArray[int32]:
    polygon = Polygon(polygon_points)
    min_x, min_y, max_x, max_y = polygon.bounds
    while True:
        # Generate random points along the major x and y axis of the polygon
        if random.choice([True, False]):
            random_x = random.uniform(min_x, max_x)
            random_y = (min_y + max_y) / 2
        else:
            random_x = (min_x + max_x) / 2
            random_y = random.uniform(min_y, max_y)
        
        random_point = Point(random_x, random_y)
        if polygon.contains(random_point):
            return np.array([int(random_point.x), int(random_point.y)], dtype=int32)

font = cv2.FONT_HERSHEY_SIMPLEX

def get_edge_points(frame: cv2.typing.MatLike, clss: list, masks: NDArray[float32]) -> list[Annotated[NDArray[int32], (2,)]]:
    """Gets the edge points of a given frame

    Returns:
        list[NDArray[int32]]: The edge points of the masks
    """
    edge_points: list[Annotated[NDArray[int32], (2,)]] = []
    for mask, cls in zip(masks, clss):
        color = colors(int(cls), True)

        # Convert the mask to integer points
        mask: NDArray[int32] = np.array(mask, dtype=int32)

        # Calculate the total perimeter of the polygon
        total_perimeter: np.float64 = np.float64(0)
        for i in range(len(mask)):
            start_point: NDArray[np.float64] = mask[i]
            end_point: NDArray[np.float64] = mask[(i + 1) % len(mask)]  # Wrap around to the first point
            total_perimeter += np.linalg.norm(start_point - end_point)

        # Spacing between points
        SPACING: int = 5

        # Interpolate points at regular intervals along the perimeter
        current_distance: np.float64 = np.float64(0)
        for i in range(len(mask)):
            start_point: NDArray[np.float64] = mask[i]
            end_point: NDArray[np.float64] = mask[(i + 1) % len(mask)]  # Wrap around to the first point
            segment_length: np.float64 = np.linalg.norm(start_point - end_point)

            while current_distance + segment_length >= SPACING:
                if (segment_length == 0):
                    print("Error: segment_length is 0. Cannot divide by 0.")
                    break

                ratio: np.float64 = (SPACING - current_distance) / segment_length
                point_float: NDArray[np.float64] = start_point + ratio * (end_point - start_point)

                # Cast point into a int32
                point: NDArray[int32] = np.round(point_float).astype(int32)
                cv2.circle(frame, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 0), thickness=-1)
                edge_points.append(point)
                start_point = point
                segment_length -= SPACING - current_distance
                current_distance = np.float64(0)

            current_distance += segment_length
    return edge_points


def hill_climb(results: list[Results], frame: cv2.typing.MatLike, center_point: NDArray[int32] | None = None, verbose: bool = False) -> None:
    if not results or results[0].masks is None or results[0].boxes is None:
        return
    
    clss = results[0].boxes.cls.tolist()
    masks = np.array(results[0].masks[0].xy) # This is actually a NDArray[NDArray[NDArray[float32]]]

    edge_points: list[Annotated[NDArray[int32], (2,)]] = get_edge_points(frame, clss, masks)
    
    center_of_mass = calculate_center_of_mass(edge_points=edge_points)
    if not Polygon(edge_points).contains(Point(center_of_mass)):
        center_of_mass = calculate_visual_center_of_polygon(edge_points=edge_points)
    
    center: Annotated[NDArray[int32], (2,)] = center_point if center_point is not None else center_of_mass
    # Hill climb to best gripper pose by drawing a line from the center of mass to the polygon edge
    closest_point: NDArray[int32] = find_closest_intersection(center=center, polygon_points=np.array(edge_points))
    magnitude: np.floating = np.linalg.norm(closest_point - center)
    
    # normalize initial
    initial: NDArray[np.floating] = (closest_point - center) / magnitude

    current = Node(direction=initial, center=center, edgepoints=edge_points)
    # Hill climb to best value for gripper position
    while True:
        new_node: Node = current.compare_neighbor()
        if np.allclose(new_node.direction, current.direction, atol=1e-6): # or (prev_direction is not None and np.allclose(new_node.direction, prev_direction, atol=1e-6)):
            break
        current: Node = new_node
        
    # conversions for print statements
    current_value: float = current.value
    VALUE_THRESHOLD: Final[int] = 20 # The threshold to use for ensuring a good gripper position
    
    if current_value <= VALUE_THRESHOLD:
        if verbose:
            print("Not optimal. Checking new center. Current value is:", current_value)
        center = get_random_point_in_polygon(np.array(edge_points))
        return hill_climb(results, frame, center, verbose=verbose)
    
    valid_gripper_length: bool = current.check_gripper_length(ratio=1.0 / max(frame.shape[0], frame.shape[1]))
    if verbose:
        (f"Valid gripper length: {valid_gripper_length}") # height, width
    if not valid_gripper_length:
        if verbose:
            print(f"Invalid gripper length. Gripper length: {current.calculate_gripper_length(ratio=1.0 / max(frame.shape[0], frame.shape[1]))}")
        center = get_random_point_in_polygon(np.array(edge_points))
        return hill_climb(results, frame, center, verbose=verbose)
    current.display(frame) # Display the gripper polygons on frame
    
    # display the midpoint value
    cv2.putText(frame, f'Midpoints in Rects: {current_value}', (10, 30), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)