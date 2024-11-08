import numpy as np
import cv2
from raycasting import find_last_intersection
from numpy.typing import NDArray
from numpy import signedinteger, int32 as _32Bit

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


def gripper_pose(direction: np.ndarray, center: np.ndarray, midpoints: NDArray[np.int32] | list[tuple[np.int32, np.int32]]):
    """
    Calculate the gripper pose based on the given direction, center, and midpoints.
    Args:
        direction (np.ndarray): The direction vector.
        center (np.ndarray): The center point.
        midpoints (list[tuple[np.int32, np.int32]]): List of midpoint coordinates.
    Returns:
        tuple: Two arrays of rectangle corner points representing the gripper pose.
    """
    
    reversedDirection = 2 * center - direction
    intersection_one = find_last_intersection(center=center, direction_pos=direction, polygon_points=np.array(midpoints))
    intersection_two = find_last_intersection(center=center, direction_pos=reversedDirection, polygon_points=np.array(midpoints))
    
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

    
    return rect_points_one, rect_points_two


def find_midpoints_in_polygon(polygon, midpoints):
    """
    Determine the number of midpoints that lie within a given polygon.
    Args:
        polygon (array-like): A list of points defining the polygon. Each point should be a tuple or list of two coordinates (x, y).
        midpoints (array-like): A list of midpoints to check. Each midpoint should be a tuple or list of two coordinates (x, y).
    Returns:
        int: The number of midpoints that are within the polygon.
    """
    
    # Number of midpoints within the rectangle
    # Check if midpoints are within the rectangles
    midpoints_within_polygon = sum(cv2.pointPolygonTest(polygon, (float(midpoint[0]), float(midpoint[1])), False) >= 0 for midpoint in midpoints)
    return midpoints_within_polygon

def display_gripper(polygons, frame):
    cv2.polylines(frame, [polygons[0]], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.polylines(frame, [polygons[1]], isClosed=True, color=(0, 0, 0), thickness=2)