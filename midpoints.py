from shapely.algorithms.polylabel import polylabel
from typing import Annotated
from deprecation import deprecated
import numpy as np
from numpy import int32
from numpy.typing import NDArray, ArrayLike
import cv2
from shapely import Polygon
from raycasting import find_last_intersection
from numpy.typing import NDArray

def get_midpoints(coordinates: np.ndarray | list[tuple]) -> list[tuple[int32, int32]]:
    """This function takes a list of coordinates and creates a line between them. It returns the coordinates of the midpoint of said line.

    Args:
        coordinates (np.ndarray | list[tuple]): A list of coordinates to get the midpoints from.

    Returns:
        list[tuple[int32, int32]]: A list of coordinates containing the midpoints of their connecting lines.
    """
    midpoints: list[tuple[int32, int32]] = []
    
    for i in range(len(coordinates) - 1):
        x1, y1 = coordinates[i]
        x2, y2 = coordinates[i + 1]
        
        midpoint = (int32((x1 + x2) / 2), int32((y1 + y2) / 2))
        midpoints.append(midpoint)
    
    return midpoints[::2] # Halfs the number of midpoints by taking every other one


def find_midpoints_in_polygon(polygon: ArrayLike, midpoints: ArrayLike) -> int:
    """
    Determine the number of midpoints that lie within a given polygon.
    Args:
        polygon (array-like): A list of points defining the polygon. Each point should be a tuple or list of two coordinates (x, y).
        midpoints (array-like): A list of midpoints to check. Each midpoint should be a tuple or list of two coordinates (x, y).
    Returns:
        int: The number of midpoints that are within the polygon.
    """

    # Ensure that polygon is an np array
    polygon = np.array(polygon)
    
    # Number of midpoints within the rectangle
    # Check if midpoints are within the rectangles
    midpoints_within_polygon: int = sum(cv2.pointPolygonTest(contour=polygon, pt=(float(midpoint[0]), float(midpoint[1])), measureDist=False) >= 0 for midpoint in midpoints) # type: ignore
    assert midpoints_within_polygon >= 0
    return midpoints_within_polygon

@deprecated(details="Use gripper pose in Node instead")
def gripper_pose(direction: np.ndarray, center: np.ndarray, midpoints: NDArray[int32] | list[tuple[int32, int32]]):
    """
    Calculate the gripper pose based on the given direction, center, and midpoints.
    Args:
        direction (np.ndarray): The direction vector.
        center (np.ndarray): The center point.
        midpoints (list[tuple[int32, int32]]): List of midpoint coordinates.
    Returns:
        tuple: Two arrays of rectangle corner points representing the gripper pose.
    """
    
    reversedDirection = -direction
    intersection_one: NDArray[int32] = find_last_intersection(center=center, direction_pos=direction, polygon_points=np.array(midpoints))
    intersection_two: NDArray[int32] = find_last_intersection(center=center, direction_pos=reversedDirection, polygon_points=np.array(midpoints))
    
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
    ], dtype=int32)
    
    # Calculate the four corners of the rectangle on two vector
    rect_points_two = np.array([
        intersection_two + rect_width / 2 * perpendicular_vector_two + rect_height / 2 * direction_vector_two,
        intersection_two - rect_width / 2 * perpendicular_vector_two + rect_height / 2 * direction_vector_two,
        intersection_two - rect_width / 2 * perpendicular_vector_two - rect_height / 2 * direction_vector_two,
        intersection_two + rect_width / 2 * perpendicular_vector_two - rect_height / 2 * direction_vector_two
    ], dtype=int32)

    
    return rect_points_one, rect_points_two

@deprecated(details="Use the display_gripper in Node instead.")
def display_gripper(polygons, frame) -> None:
    cv2.polylines(frame, [polygons[0]], isClosed=True, color=(0, 0, 0), thickness=2)
    cv2.polylines(frame, [polygons[1]], isClosed=True, color=(0, 0, 0), thickness=2)

def calculate_center_of_mass(edge_points) -> Annotated[NDArray[int32], (2,)]:
    """Calculates the center of mass using the moments of edge_points

    Args:
        edge_points (array-like): The edge_points to get the center of mass from

    Returns:
        NDArray[int32]: The center of mass coordinates.
    
    Examples:
        >>> center = calculate_center_of_mass(edge_points=edge_points)
        >>> cv2.circle(frame, tuple(center), 5, (255, 0, 0), -1) # Draw the center of mass
    """
    moments: cv2.typing.Moments = cv2.moments(np.array(edge_points, dtype=int32))
    # Compute the center of mass
    if moments["m00"] != 0:
        cX: int = int(moments["m10"] / moments["m00"])
        cY: int = int(moments["m01"] / moments["m00"])
    else:
        cX, cY = 0, 0
    return np.array([cX, cY], dtype=int32)

def calculate_visual_center_of_polygon(edge_points) -> Annotated[NDArray[int32], (2,)]:
    """Finds the visual center of a polygon using the polylable library.

    Args:
        polygon (array-like): The polygon to find the visual center of. Each point should be a tuple or list of two coordinates (x, y).

    Returns:
        NDArray[int32]: The visual center coordinates of the polygon.
    """

    polygon = Polygon(edge_points)
    label_point = polylabel(polygon)
    return np.array([int32(label_point.x), int32(label_point.y)], dtype=int32)