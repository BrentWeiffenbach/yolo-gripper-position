import cv2
import numpy as np
from numpy import int32, float32
from numpy.typing import NDArray
from typing import Annotated, Final
from deprecation import deprecated
from midpoints import find_midpoints_in_polygon
from raycasting import find_last_intersection
from math import floor

class Node:
    # Define the rectangle size
    RECT_WIDTH: int = 60
    RECT_HEIGHT: int = 8
    MAX_GRIPPER_LENGTH: Final[int] = 30 # Arbitrary units
    
    def __init__(self, direction: Annotated[NDArray[float32], (2,)], center: Annotated[NDArray[int32], (2,)], edgepoints: list[NDArray[int32]]) -> None:
        """Constructor for Node

        Args:
            direction (NDArray[float32]): Current gripper direction as a unit vector
            center (NDArray[int32]): Center of the gripper
            edgepoints (list[NDArray[int32]]): list of edgepoints to evaluate
        """

        self.direction = direction
        self.center = center
        self.edgepoints: list[NDArray[int32]] = edgepoints

        # Default RECT_HEIGHT and RECT_WIDTH with values
        if Node.RECT_HEIGHT is None or Node.RECT_WIDTH is None:
            Node.RECT_HEIGHT = 6
            Node.RECT_WIDTH = 60

        # Calculate the gripper's polygons
        self.gripper_polygons: tuple[NDArray[int32], NDArray[int32]] = self.calculate_gripper_polygons()

        # Calculate the number of edgepoints within the gripper's polygons
        self.value: float = self.calculate_value()

    @staticmethod
    @deprecated("Image size is resized in clamp_frame")
    def set_rect_size(frame: cv2.typing.MatLike) -> tuple[int, int]:
        _RECT_WIDTH: Final[int] = 60
        _RECT_HEIGHT: Final[int] = 6
        height, width = frame.shape[:2]
        ref_width, ref_height = (640, 640)

        width_scale = width / ref_width
        height_scale = height / ref_height

        _width = int(_RECT_WIDTH * width_scale)
        _height = int(_RECT_HEIGHT * height_scale)
        
        Node.RECT_WIDTH = _width
        Node.RECT_HEIGHT = _height
        return Node.RECT_WIDTH, Node.RECT_HEIGHT

    def rotate(self, angle_in_deg: int) -> NDArray[float32]:
        radians: float = -angle_in_deg * np.pi / 180

        rotation_matrix: NDArray[float32] = np.array([[np.cos(radians), np.sin(radians)],
                                    [-np.sin(radians), np.cos(radians)]],
                                    dtype=float32)
        neighbor: NDArray[float32] = np.dot(rotation_matrix, self.direction)

        return neighbor / np.linalg.norm(neighbor)

    def gripper_pose(self) -> tuple[NDArray[int32], NDArray[int32]]:
        """Calculate the gripper pose based on the node's direction, center, and edge points.

        Returns:
            (tuple[NDArray[int32], NDArray[int32]]): Two arrays of rectangle corner points representing the gripper pose.
        """
         
        reversedDirection = -self.direction
        intersection_one: NDArray[int32] = find_last_intersection(center=self.center, direction_pos=self.direction, polygon_points=np.array(self.edgepoints))
        intersection_two: NDArray[int32] = find_last_intersection(center=self.center, direction_pos=reversedDirection, polygon_points=np.array(self.edgepoints))
        
        # Calculate the direction vector of the intersection line
        direction_vector_one = intersection_one - self.center
        direction_vector_two = intersection_two - self.center
        direction_vector_one = direction_vector_one / np.linalg.norm(direction_vector_one)  # Normalize the vector
        direction_vector_two = direction_vector_two / np.linalg.norm(direction_vector_two)  # Normalize the vector

        # Calculate the perpendicular vector
        perpendicular_vector_one = np.array([-direction_vector_one[1], direction_vector_one[0]])
        perpendicular_vector_two = np.array([-direction_vector_two[1], direction_vector_two[0]])
        
        # Calculate the four corners of the rectangle on one vector 
        rect_points_one: NDArray[int32] = np.array([
            intersection_one + self.RECT_WIDTH / 2 * perpendicular_vector_one - self.RECT_HEIGHT / 2 * direction_vector_one,
            intersection_one - self.RECT_WIDTH / 2 * perpendicular_vector_one - self.RECT_HEIGHT / 2 * direction_vector_one,
            intersection_one - self.RECT_WIDTH / 2 * perpendicular_vector_one + self.RECT_HEIGHT / 2 * direction_vector_one,
            intersection_one + self.RECT_WIDTH / 2 * perpendicular_vector_one + self.RECT_HEIGHT / 2 * direction_vector_one
        ], dtype=int32) # MUST be int32 or an assertion fails
        
        # Calculate the four corners of the rectangle on two vector
        rect_points_two: NDArray[int32] = np.array([
            intersection_two + self.RECT_WIDTH / 2 * perpendicular_vector_two - self.RECT_HEIGHT / 2 * direction_vector_two,
            intersection_two - self.RECT_WIDTH / 2 * perpendicular_vector_two - self.RECT_HEIGHT / 2 * direction_vector_two,
            intersection_two - self.RECT_WIDTH / 2 * perpendicular_vector_two + self.RECT_HEIGHT / 2 * direction_vector_two,
            intersection_two + self.RECT_WIDTH / 2 * perpendicular_vector_two + self.RECT_HEIGHT / 2 * direction_vector_two
        ], dtype=int32) # MUST be int32 or an assertion fails

        return rect_points_one, rect_points_two


    def calculate_gripper_polygons(self) -> tuple[NDArray[int32], NDArray[int32]]:
        """Calculate the polygons for the gripper at its current position"""
        return self.gripper_pose()

    def get_midpoints_in_node(self) -> int:
        """Calculate the number of edgepoints in the gripper at its current position

        Returns:
            int: The number of edgepoints in the node's polygons
        """
        p1, p2 = self.gripper_polygons

        return find_midpoints_in_polygon(p1, self.edgepoints) + find_midpoints_in_polygon(p2, self.edgepoints)

    def calculate_value(self) -> float:
        """Calculates the number of midpoints and subtracts the intersection area of the gripper from them

        Returns:
            float: number of midpoints - (intersection area of gripper_polygons[0] + intersection area of gripper_polygons[1])
        """
        _value: int = self.get_midpoints_in_node()
        assert _value >= 0 # Assert to ensure that value is never less than 0
        return _value
    
    def find_neighbor(self, angle_in_deg: int) -> 'Node':
        """Finds the neighbor in the given degrees

        Args:
            angle_in_deg (int): The degrees (cw) to check for neighbors. Negative degrees represents a ccw neighbor.

        Returns:
            Node: A new instance of Node representing the neighbor
        
        Examples:
            >>> cw_node = my_node.find_neighbor(10)
            Node
            >>> ccw_node = my_node.find_neighbor(-10)
            Node
        """
        direction = self.rotate(angle_in_deg)
        
        return Node(direction=direction, center=self.center, edgepoints=self.edgepoints)

    def compare_neighbor(self, angle_in_deg: int=5) -> 'Node':
        """Compares the clockwise and counter clockwise neighbors to this node

        Args:
            angle_in_deg (int, optional): The angle in degrees to check for neighbors in. Defaults to 10.

        Returns:
            Node: The node with the greater value
        """
        cw_node: Node = self.find_neighbor(angle_in_deg=angle_in_deg)
        ccw_node: Node = self.find_neighbor(angle_in_deg=-angle_in_deg)

        max_neighbor: float = max(self.value, cw_node.value, ccw_node.value)
        if max_neighbor > self.value:
            if max_neighbor == cw_node.value:
                return cw_node
            if max_neighbor == ccw_node.value:
                return ccw_node
        return self
    
    def display(self, frame: cv2.typing.MatLike, color: tuple[int, int, int] = (0, 255, 255)) -> None:
        """Displays the gripper to frame

        Args:
            frame (cv2.typing.MatLike): The frame to display the gripper to
            color (tuple[int, int, int]): The color of the grippers to display. Defaults to (0, 255, 255)
        """
        p1, p2 = self.gripper_polygons
        cv2.polylines(frame, [p1], isClosed=True, color=color, thickness=2)
        cv2.polylines(frame, [p2], isClosed=True, color=color, thickness=2)

        # Draw line between grippers
        p1_mean = np.mean(p1, axis=0).astype(int)
        p2_mean = np.mean(p2, axis=0).astype(int)

        cv2.line(frame, tuple(p1_mean), tuple(p2_mean), color=(255, 0, 0), thickness=2)

    def calculate_gripper_length(self, ratio: float) -> np.floating:
        p1, p2 = self.gripper_polygons
        p1_mean = np.mean(p1, axis=0)
        p2_mean = np.mean(p2, axis=0)
        _distance = np.linalg.norm(p1_mean - p2_mean)
        return _distance * ratio

    def check_gripper_length(self, ratio: float) -> bool:
        # Calculate distance between gripper polygons
        _gripper_distance = self.calculate_gripper_length(ratio)
        return floor(_gripper_distance * 100) <= Node.MAX_GRIPPER_LENGTH

@staticmethod
@deprecated(details="Does not work and is not used, could be used to add another heuristic")
def calculate_intersection_area(polygon1: NDArray[int32], polygon2: NDArray[int32]) -> float:
    """Calculate the intersection area between two polygons."""
    try:
        intersection, _ = cv2.intersectConvexConvex(np.array(polygon1, dtype=float32), np.array(polygon2, dtype=float32))
        return intersection
    except Exception as e:
        print("Error trying to calculate the intersection area: ", e)
        return 0.0 # Defults to 0.0

@staticmethod
@deprecated(details="Would be used with caluclate_intersection_area")
def get_minimum_bounding_box(edge_points: list[Annotated[NDArray[int32], (2,)]]) -> Annotated[NDArray[int32], (4,)]:
    """Finds the minimum bounding box of all the edge points. 

    Args:
        edge_points (list[NDArray[int32]]): The edge points to get the bounding box from.

    Returns:
        NDArray[int32]: A bounding box of the edge points. An np array of 4 points.
    """
    points = np.array(edge_points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])