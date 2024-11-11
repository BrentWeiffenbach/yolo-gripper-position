import cv2
import numpy as np
import numpy.typing as npt
from typing import Annotated, List

from midpoints import find_midpoints_in_polygon
from raycasting import find_last_intersection

class Node:
    # Define the rectangle size
    RECT_WIDTH = 60
    RECT_HEIGHT = 10
    
    def __init__(self, direction: Annotated[npt.NDArray[np.float32], (2,)], center: Annotated[npt.NDArray[np.int32], (2,)], edgepoints: List[npt.NDArray[np.int32]]):
        """Constructor for Node

        Args:
            direction (npt.NDArray[np.float32]): Current gripper direction as a unit vector
            center (npt.NDArray[np.int32]): Center of the gripper
            edgepoints (List[npt.NDArray[np.int32]]): List of edgepoints to evaluate
        """

        self.direction = direction
        self.center = center
        # Removed this code from the typedef of edgepoints:
        # | list[tuple[np.int32, np.int32]]
        self.edgepoints = edgepoints

        # Calculate the gripper's polygons
        self.gripper_polygons: tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]] = self.calculate_gripper_polygons()

        # Calculate the number of edgepoints within the gripper's polygons
        self.value: float = self.calculate_value()

    def rotate(self, angle_in_deg: int) -> npt.NDArray[np.float32]:
        radians = angle_in_deg * np.pi / 180

        rotation_matrix = np.array([[np.cos(radians), np.sin(radians)],
                                    [-np.sin(radians), np.cos(radians)]],
                                    dtype=np.float32)
        neighbor: npt.NDArray[np.float32] = np.dot(rotation_matrix, self.direction)

        return neighbor / np.linalg.norm(neighbor)

    def gripper_pose(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Calculate the gripper pose based on the node's direction, center, and edge points.

        Returns:
            (tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]): Two arrays of rectangle corner points representing the gripper pose.
        """
         
        reversedDirection = -self.direction
        intersection_one: npt.NDArray[np.float64] = find_last_intersection(center=self.center, direction_pos=self.direction, polygon_points=np.array(self.edgepoints))
        intersection_two: npt.NDArray[np.float64] = find_last_intersection(center=self.center, direction_pos=reversedDirection, polygon_points=np.array(self.edgepoints))
        
        # Calculate the direction vector of the intersection line
        direction_vector_one = intersection_one - self.center
        direction_vector_two = intersection_two - self.center
        direction_vector_one = direction_vector_one / np.linalg.norm(direction_vector_one)  # Normalize the vector
        direction_vector_two = direction_vector_two / np.linalg.norm(direction_vector_two)  # Normalize the vector

        # Calculate the perpendicular vector
        perpendicular_vector_one = np.array([-direction_vector_one[1], direction_vector_one[0]])
        perpendicular_vector_two = np.array([-direction_vector_two[1], direction_vector_two[0]])
        
        # Calculate the four corners of the rectangle on one vector 
        rect_points_one = np.array([
            intersection_one + self.RECT_WIDTH / 2 * perpendicular_vector_one + self.RECT_HEIGHT / 2 * direction_vector_one,
            intersection_one - self.RECT_WIDTH / 2 * perpendicular_vector_one + self.RECT_HEIGHT / 2 * direction_vector_one,
            intersection_one - self.RECT_WIDTH / 2 * perpendicular_vector_one - self.RECT_HEIGHT / 2 * direction_vector_one,
            intersection_one + self.RECT_WIDTH / 2 * perpendicular_vector_one - self.RECT_HEIGHT / 2 * direction_vector_one
        ], dtype=np.int32) # MUST be np.int32 or an assertion fails
        
        # Calculate the four corners of the rectangle on two vector
        rect_points_two = np.array([
            intersection_two + self.RECT_WIDTH / 2 * perpendicular_vector_two + self.RECT_HEIGHT / 2 * direction_vector_two,
            intersection_two - self.RECT_WIDTH / 2 * perpendicular_vector_two + self.RECT_HEIGHT / 2 * direction_vector_two,
            intersection_two - self.RECT_WIDTH / 2 * perpendicular_vector_two - self.RECT_HEIGHT / 2 * direction_vector_two,
            intersection_two + self.RECT_WIDTH / 2 * perpendicular_vector_two - self.RECT_HEIGHT / 2 * direction_vector_two
        ], dtype=np.int32) # MUST be np.int32 or an assertion fails

        
        return rect_points_one, rect_points_two


    def calculate_gripper_polygons(self) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """Calculate the polygons for the gripper at its current position
        """
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
        p1, p2 = self.gripper_polygons
        _value = self.get_midpoints_in_node()
        # return _value - calculate_intersection_area(p1, self.edgepoints) + calculate_intersection_area(p2, self.edgepoints)
        new_value = _value - calculate_intersection_area(p1, get_minimum_bounding_box(self.edgepoints)) + calculate_intersection_area(p2, get_minimum_bounding_box(self.edgepoints))
        assert new_value >= 0 # Assert to ensure that value is never less than 0
        return new_value
    
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
        # print(type(direction[0]))
        
        return Node(direction=direction, center=self.center, edgepoints=self.edgepoints)

    def compare_neighbor(self, angle_in_deg: int=10) -> 'Node':
        """Compares the clockwise and counter clockwise neighbors to thise node

        Args:
            angle_in_deg (int, optional): The angle in degrees to check for neighbors in. Defaults to 10.

        Returns:
            Node: The node with the greater value
        """
        cw_node = self.find_neighbor(angle_in_deg=angle_in_deg)
        ccw_node = self.find_neighbor(angle_in_deg=-angle_in_deg)

        max_neighbor = max(self.value, cw_node.value, ccw_node.value)
        if max_neighbor > self.value:
            if max_neighbor == cw_node.value:
                print("moving cw from current:", self.value, " to: ", cw_node.value, ", Where direction current: ", self.direction, ", new direction: ", cw_node.direction)
                return cw_node
            if max_neighbor == ccw_node.value:
                print("moving ccw from current:", self.value, " to: ", ccw_node.value, ", Where direction current: ", self.direction, ", new direction: ", ccw_node.direction)
                return ccw_node
        print("found max at: ", max_neighbor)
        return self
    
    def display(self, frame: cv2.typing.MatLike) -> None:
        """Displays the gripper to frame

        Args:
            frame (cv2.typing.MatLike): The frame to display the gripper to
        """
        p1, p2 = self.gripper_polygons
        cv2.polylines(frame, [p1], isClosed=True, color=(0, 0, 0), thickness=2)
        cv2.polylines(frame, [p2], isClosed=True, color=(0, 0, 0), thickness=2)

@staticmethod
def calculate_intersection_area(polygon1: npt.NDArray[np.int32], polygon2: npt.NDArray[np.int32]) -> float:
    """Calculate the intersection area between two polygons."""
    intersection, _ = cv2.intersectConvexConvex(np.array(polygon1, dtype=np.float32), np.array(polygon2, dtype=np.float32))
    return intersection or 0.0

@staticmethod
def get_minimum_bounding_box(edge_points: List[Annotated[npt.NDArray[np.int32], (2,)]]) -> Annotated[npt.NDArray[np.int32], (4,)]:
    """Finds the minimum bounding box of all the edge points. 

    Args:
        edge_points (List[npt.NDArray[np.int32]]): The edge points to get the bounding box from.

    Returns:
        npt.NDArray[np.int32]: A bounding box of the edge points. An np array of 4 points.
    """
    points = np.array(edge_points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    return np.array([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])

        

