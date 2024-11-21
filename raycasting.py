import numpy as np
import numpy.typing as npt
from typing import Annotated, Optional, List
from numba import njit
from numba.np.extensions import cross2d

@njit
def ray_math(v1, v2, v3):
    v1 = v1.astype(np.float64)
    v2 = v2.astype(np.float64)
    v3 = v3.astype(np.float64)
    dot = np.dot(v2, v3)
    if abs(dot) < 1e-6:
        return
    t1 = cross2d(v2, v1) / dot
    t2 = np.dot(v1, v3) / dot
    # if t1 >= 0 and 0 <= t2 <= 1:
    #     return ray_origin + t1 * ray_direction
    # return None
    if t1 >= 0 and 0 <= t2 <= 1:
        return t1

@njit
def intersect(ray_origin: np.ndarray, ray_direction: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculates intersection between a ray and polygon segment
    
    Args:
        ray_origin (np.ndarray): Start of ray
        ray_direction (np.ndarray): direction of ray
        segment_start (np.ndarray): start of polygon segment
        segment_end (np.ndarray): end of polygon segment
    
    Returns:
        Optional[np.ndarray]: Intersection point if ray intersects segment
    """
    # segment_start = np.array(segment_start)
    # segment_end = np.array(segment_end)
    v1 = ray_origin - segment_start
    v2 = segment_end - segment_start
    # perpendicular vector to the ray's direction
    v3 = np.array([-ray_direction[1], ray_direction[0]], dtype=np.float64)

    ray_math_result = ray_math(v1, v2, v3)
    
    return  ray_origin + ray_math_result * ray_direction if ray_math_result is not None else None

def find_last_intersection(center: Annotated[npt.NDArray[np.int32], (2,)], direction_pos: Annotated[npt.NDArray[np.float32], (2,)], polygon_points: npt.NDArray[np.int32]) -> npt.NDArray[np.float32]:
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
    intersections: List[np.ndarray] = []
    
    for i in range(len(polygon_points)):
        segment_start = polygon_points[i]
        segment_end = polygon_points[(i+1) % len(polygon_points)]
        intersection = intersect(center, ray_direction, np.array(segment_start), np.array(segment_end))
        if intersection is not None:
            intersections.append(intersection)
    
    if intersections:
        distances = [np.linalg.norm(point - center) for point in intersections]
        return intersections[np.argmax(distances)]
    # found no intersection
    # print("found no intersection")
    return np.ndarray([])

def find_closest_intersection(center: np.ndarray, polygon_points: np.ndarray) -> np.ndarray:
    # init inersections
    intersections: List[np.ndarray] = []
    distances: List[float] = []
    
    # loop through polygon points to find every intersection
    for i in range(len(polygon_points)):
        segment_start = polygon_points[i]
        segment_end = polygon_points[(i+1) % len(polygon_points)]
        ray_direction = polygon_points[i] - center
        intersection = intersect(center, ray_direction, segment_start, segment_end)
        if intersection is not None:
            # print("found intersection at", intersection)=-1)
            distance = np.linalg.norm(intersection - center)
            distance = float(distance)
            distances.append(distance)
            # print("distance is", distance)
            intersections.append(intersection)
    if intersections:
        min_distance_index = np.argmin(distances)
        min_distance = distances[min_distance_index]
        closest_intersection = intersections[min_distance_index]
        # print(f"Lowest distance: {min_distance}, Intersection point: {closest_intersection}")
        return closest_intersection
    return np.ndarray([])