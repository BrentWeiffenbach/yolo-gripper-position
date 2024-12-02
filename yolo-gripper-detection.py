import sys
import cv2
import os
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.model import Results
from yolo_utils.setup_yolo import setup_yolo
from typing import Literal, NoReturn
from collections.abc import Generator
from hillclimb import hill_climb

# Ask user if it should use a video file or webcam
def get_setup_type() -> Literal['webcam', 'video', 'photo']:
    print("Would you like to grab objects on webcam, video, or a photo file?")
    while True:
        setup_type: Literal['webcam', 'video', 'photo'] | str  = input("Enter 'webcam', 'video', or 'photo': ").strip().lower()
        if not setup_type:
            setup_type = 'webcam'
        if setup_type not in ['webcam', 'video', 'photo']:
            print("Invalid input. Please enter 'webcam', 'video', or 'photo'")
            continue # Rereun if it is not valid
        break
    assert setup_type in ['webcam', 'video', 'photo'] # Ensure that it is in the allowed types
    return setup_type # type: ignore

def find_file(directory: str, file_name: str) -> str:
    for root, _, files in os.walk(directory):
        if file_name in files:
            return os.path.join(root, file_name)
    return ""

def check_file_type(path: str, allowed_extensions: list[str]) -> bool:
    """Checks if the file type is in the allowed file type extensions.

    Args:
        path (str): The file to check.
        allowed_extentions (list[str]): The allowed extensions for the file type.

    Returns:
        bool: A flag indicating if the path is an allowed file type.

    Example:
        ```
        allowed_file_types: list[str] = [".jpg", ".jpeg", ".png"]
        allowed_file: bool = check_file_type("my_file.jpg", allowed_file_types) # True
        disallowed_file: bool = check_file_type("my_file.mp4", allowed_file_types) # False
        ```
    """
    _, extension = os.path.split(path)
    return extension.split('.')[-1].lower() in allowed_extensions

def check_file_extension_valid(file_path: str, setup_type: Literal["video", "photo", "webcam"]) -> bool:
    """Checks if a file extension is valid. There are lists containing valid file extensions in the body of this function.

    Args:
        file_path (str): The file path to check.
        setup_type ("video", "photo", or "webcam"): The setup_type.

    Returns:
        bool: True if the extension is valid. Will exit the program if it is not.
    """
    assert (setup_type == "photo" and cv2.haveImageReader(filename=file_path)) or True # Ensure that cv2 can read the file
    assert setup_type != "webcam" # Ensure that a webcam is not passed in here

    # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#imread
    SUPPORTED_PHOTO_EXTS: list[str] = ["bmp", "dib", "jpg", "jpeg", "jpe", "jp2", "png", "pbm", "pgm", "ppm", "sr", "ras", "tiff", "tif"]
    # https://fourcc.org/codecs.php
    SUPPORTED_VIDEO_EXTS: list[str] = ["mp4", "mp4v"]

    if not check_file_type(file_path, SUPPORTED_PHOTO_EXTS if setup_type == "photo" else SUPPORTED_VIDEO_EXTS if setup_type == "video" else []):
        sys.exit(f"Unsupported file type for the {setup_type.capitalize()} yolo gripper demo!")
    return True

def get_file_path(setup_type: Literal["video", "photo", "webcam"]) -> str:
    """Gets a file path. Will exit if the file path is not found.

    Returns:
        str: The file path to read from
    """
    if setup_type == "webcam":
        return ""
    
    # Path to the downloaded video
    file_name: str = input(f"Enter the name of the file to grab objects on (default is example.{'jpg' if setup_type == 'photo' else 'mp4'}): ").strip()
    
    if not file_name:
        print("No file name provided, going with default.")
        if setup_type == 'video':
            file_name = 'example.mp4'
        elif setup_type == 'photo':
            file_name = 'example.jpg'
        else:
            print("Unsupported file type!")
            exit(1)
    file_path: str = find_file(os.getcwd(), file_name)
    
    # Ensure that the file at file_path exists
    if not os.path.isfile(file_path):
        print(f"Absolute path: {file_path}")
        print(f"Could not find file: {file_path}")
        exit(1)
    
    assert file_path # Ensure that file_path is not Unknown
    assert check_file_extension_valid(file_path=file_path, setup_type=setup_type)
    return file_path

class YoloGripperDetection():
    """The main class for yolo gripper detection.

    Example:
        >>> # Display an image from file
        yolo = YoloGripperDetection(setup_type='photo')
        yolo.display(path='example.jpg')

        >>> # Display a video from file
        yolo = YoloGripperDetection(setup_type='video')
        yolo.display(path='example.mp4')

        >>> # Ask user for file and display video
        yolo = YoloGripperDetection(setup_type='video')
        yolo.display()

        >>> # Ask user for file and export video
        yolo = YoloGripperDetections(setup_type='video')
        yolo.export(file_name='example_annotated.mp4')
    """
    MODEL_NAME = 'yolo11s-seg.pt'
    # 0 is person
    # frisbee, sports ball, bottle, cup, fork, knife, spoon, banana, apple, orange, carrot, mouse, remote, cell phone, book, vase, scissors, toothbrush
    TRACKED_CLASSES: list[int] = [29, 39, 41, 42, 43, 44, 46, 47, 49, 51, 64, 65, 67, 73, 75, 76, 79]

    font: int = cv2.FONT_HERSHEY_SIMPLEX
    setup_yolo(MODEL_NAME) # Download YOLO if it doesn't exist
    model: Model = YOLO(MODEL_NAME)

    def __init__(self, setup_type: Literal['webcam', 'video', 'photo']) -> None:
        self.setup_type: Literal['webcam', 'video', 'photo'] = setup_type

        # Ensure that setup_type is a valid type
        if self.setup_type not in ['webcam', 'video', 'photo']:
            self.setup_type = get_setup_type()

        self.camera_source: int = 0
    
    def __call__(self, path: str | None = None) -> None:
        """Calls `display()`.

        Args:
            path (str, optional): The file path to read from. None if it is a webcam. Defaults to None.
        """
        self.display(path=path)

    def detect(self, source_frame: cv2.typing.MatLike) -> tuple[list[Results], list[str]]:
        results: list[Results] = YoloGripperDetection.model.track(source_frame, classes=YoloGripperDetection.TRACKED_CLASSES)
        hill_climb(results, source_frame)
        
        # Get the detected classes
        detected_classes: list[str] = []
        if results[0].boxes is not None:
            detected_classes = [YoloGripperDetection.model.names[int(cls)] for cls in results[0].boxes.cls]
        
        return results, detected_classes

    def photo_detection(self, path: str) -> tuple[cv2.typing.MatLike, list[str]]:
        """Runs the hillclimbing algorithm on `path`.

        Args:
            path (str, optional): The path to get the photo from. Will ask user for input if path is `None`. Defaults to `None`.

        Returns:
            tuple (cv2.typing.MatLike, list[str]): A tuple containing the `frame` and `detected_classes`.
        
        Example:
            >>> frame, detected_classes = self.photo_detection(path)
        """
        assert self.setup_type == "photo"
        file_path: str = path
        frame: cv2.typing.MatLike = cv2.imread(file_path) # TODO: Does not actually check if this is a valid path
        
        results, detected_classes = self.detect(frame)
        class_text: str = ', '.join(detected_classes)
        cv2.putText(frame, text=f"Detected: {class_text}", org=(frame.shape[0], frame.shape[0] - 10), fontFace=YoloGripperDetection.font, fontScale=0.5, color=(0, 0, 255), thickness=1)

        return frame, detected_classes
    
    def video_detection(self, path: str | None = None) -> Generator[tuple[cv2.typing.MatLike, list[str]], None, None]:
        # Ensure only those with a cap are availible at this point
        assert self.setup_type == 'webcam' or self.setup_type == 'video'
        assert (self.setup_type == 'video' and path is not None) or (self.setup_type == 'webcam' and path is None)

        # Gets the source for the VideoCapture. Either a path to a file, or the webcam camera_source
        CAPTURE_SOURCE: str | int | None = path if self.setup_type == 'video' else self.camera_source

        assert CAPTURE_SOURCE is not None # None was added to the typing to get rid of error, but it will never be there
        
        cap: cv2.VideoCapture = cv2.VideoCapture(CAPTURE_SOURCE)
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
            if not success:
                break

            # frame = cv2.resize(frame, (640, 640))
            # frame = frame[:-100, :]  # Crop the bottom 30 pixels
            results, detected_classes = self.detect(frame)
            class_text: str = ', '.join(detected_classes)
            cv2.putText(frame, text=f"Detected: {class_text}", org=(0, frame.shape[0] - 50), fontFace=YoloGripperDetection.font, fontScale=0.5, color=(0, 0, 255), thickness=1)
            
            yield frame, detected_classes
            
            # Check to see if there is a pressed key
            key: int = cv2.waitKey(1) & 0xFF

            # Break the loop if 'q' is pressed
            if key == ord("q"):
                break

            if self.setup_type == 'webcam':
                # Switch camera if 'c' is pressed
                if key & 0xFF == ord("c"):
                    self.camera_source = 1 - self.camera_source # type: ignore
                    if cap.isOpened():
                        cap.release()
                        
                    cap = cv2.VideoCapture(self.camera_source)

        cap.release()

    def __display_photo(self, path: str) -> NoReturn:
        """Displays the photo detected from `photo_detection`.

        Args:
            path (str, optional): The path to the photo to display. Will ask user for input if it is `None`. Defaults to `None`.

        Returns:
            NoReturn: Does not return anything. Will exit the program.
        """
        assert self.setup_type == "photo"

        frame, detected_classes = self.photo_detection(path)

        # Add information to quit to frame
        cv2.putText(frame, text="Press any key to quit", org=(0, frame.shape[0] - 10), fontFace=YoloGripperDetection.font, fontScale=0.5, color=(0, 0, 255))

        # Display the annotated frame
        cv2.imshow("YOLO Inference", frame)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()
        exit(1)
    
    def __display_video(self, path: str | None = None) -> None:
        for frame, detected_classes in self.video_detection(path):
            # Display the keybind information
            # Note that this logic is separate from the actual key handling. This is because it needs to change cap.
            # Not good for maintainability.
            cv2.putText(frame, text="Press q to quit", org=(0, frame.shape[0] - 10), fontFace=YoloGripperDetection.font, fontScale=0.5, color=(0, 0, 255))

            if self.setup_type == 'webcam':
                cv2.putText(frame, text="Press c to switch webcam source", org=(0, frame.shape[0] - 30), fontFace=YoloGripperDetection.font, fontScale=0.5, color=(0, 0, 255))
            
            cv2.imshow("Gripper", frame)
        cv2.destroyAllWindows()

    def display(self, path: str | None = None) -> None:
        """Displays the webcam/file at `path`.

        Args:
            path (str, optional): The path to the photo/video to display. Will ask user for input if it is `None`. Defaults to `None`.
        """

        _path: str = path or get_file_path(self.setup_type)
        if self.setup_type not in ["webcam"]:
            # Ensure that the user isn't passing an invalid file type
            # e.g. a jpg for a video
            check_file_extension_valid(_path, self.setup_type)
        if self.setup_type == "photo":
            self.__display_photo(_path)
        if self.setup_type == "video":
            self.__display_video(_path)
        if self.setup_type == "webcam":
            self.__display_video()

yolo = YoloGripperDetection(setup_type='video')
yolo()