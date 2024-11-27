import numpy as np
import numpy.typing as npt
from numpy import int32, float32, float64
from numpy.typing import NDArray
from typing import Annotated
from numba import njit
from numba.np.extensions import cross2d

@njit
def ray_math(v1, v2, v3):
    v1 = v1.astype(float64)
    v2 = v2.astype(float64)
    v3 = v3.astype(float64)
    dot = np.dot(v2, v3)
    if abs(dot) < 1e-6:
        return
    t1 = cross2d(v2, v1) / dot
    t2 = np.dot(v1, v3) / dot
    if t1 >= 0 and 0 <= t2 <= 1:
        return t1

@njit
def intersect(ray_origin: np.ndarray, ray_direction: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray) -> np.ndarray | None:
    """
    Calculates intersection between a ray and polygon segment
    
    Args:
        ray_origin (np.ndarray): Start of ray
        ray_direction (np.ndarray): direction of ray
        segment_start (np.ndarray): start of polygon segment
        segment_end (np.ndarray): end of polygon segment
    
    Returns:
        (np.ndarray | None): Intersection point if ray intersects segment
    """
    v1 = ray_origin - segment_start
    v2 = segment_end - segment_start
    # perpendicular vector to the ray's direction
    v3: NDArray[float64] = np.array([-ray_direction[1], ray_direction[0]], dtype=float64)
    # do math to get t1 or nothing 
    # t1 is the cross product of v2, v1 / the dot product of v2 and v3
    ray_math_result = ray_math(v1, v2, v3)
    
    return  ray_origin + ray_math_result * ray_direction if ray_math_result is not None else None

def find_last_intersection(center: Annotated[NDArray[int32], (2,)], direction_pos: Annotated[NDArray[float32], (2,)], polygon_points: NDArray[int32]) -> NDArray[int32]:
    """
    Finds last intersection point along ray on polygon
    
    Args:
        center (np.ndarray): center of mass for polygon
        direction_pos (np.ndarray): direction of vector towards gripper pose
        polygon_points (np.ndarray): list of polygon vertices as a 2d array
    
    Returns:
        np.ndarray: Last intersection point if found
    """
    ray_direction = direction_pos
    intersections: list[np.ndarray] = []
    
    for i in range(len(polygon_points)):
        segment_start: NDArray[int32] = polygon_points[i]
        segment_end: NDArray[int32] = polygon_points[(i+1) % len(polygon_points)]
        intersection = intersect(center, ray_direction, np.array(segment_start), np.array(segment_end))
        if intersection is not None:
            intersections.append(intersection)
    
    if intersections:
        distances = [np.linalg.norm(point - center) for point in intersections]
        return intersections[np.argmax(distances)]
    # found no intersection
    return np.ndarray([])

def find_closest_intersection(center: np.ndarray, polygon_points: np.ndarray) -> np.ndarray:
    """Finds the closest intersection of polygon_points from center using ray casting.

    Args:
        center (np.ndarray): The center of mass to start the rays from
        polygon_points (np.ndarray): The polygon points to get the segments from.

    Returns:
        np.ndarray: The point of closest intersection, or an empty array if there is no intersection.
    """
    # init intersections
    intersections: list[np.ndarray] = []
    distances: list[float] = []
    
    # loop through polygon points to find every intersection
    for i in range(len(polygon_points)):
        segment_start: NDArray[int32] = polygon_points[i]
        segment_end: NDArray[int32] = polygon_points[(i+1) % len(polygon_points)]
        ray_direction = polygon_points[i] - center
        intersection = intersect(center, ray_direction, segment_start, segment_end)
        if intersection is not None:
            distance = np.linalg.norm(intersection - center)
            distance = float(distance)
            distances.append(distance)
            intersections.append(intersection)
    if intersections:
        min_distance_index = np.argmin(distances)
        closest_intersection = intersections[min_distance_index]
        return closest_intersection
    return np.ndarray([])