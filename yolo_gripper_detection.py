import sys
import cv2
import os
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.model import Model
from ultralytics.engine.model import Results
from yolo_utils.setup_yolo import setup_yolo
from typing import Final, Literal, NoReturn
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
    assert (setup_type == "photo" and os.path.exists(path=file_path) and cv2.haveImageReader(filename=file_path)) or True # Ensure that cv2 can read the file

    # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html?highlight=imread#imread
    SUPPORTED_PHOTO_EXTS: Final[list[str]] = ["bmp", "dib", "jpg", "jpeg", "jpe", "jp2", "png", "pbm", "pgm", "ppm", "sr", "ras", "tiff", "tif"]
    # https://fourcc.org/codecs.php
    SUPPORTED_VIDEO_EXTS: Final[list[str]] = ["mp4", "mp4v"]

    _PHOTO_TYPES: Final[list[str]] = ["photo"]
    _VIDEO_TYPES: Final[list[str]] = ["video", "webcam"]

    if not check_file_type(file_path, SUPPORTED_PHOTO_EXTS if setup_type in _PHOTO_TYPES else SUPPORTED_VIDEO_EXTS if setup_type in _VIDEO_TYPES else []):
        sys.exit(f"Unsupported file type for the {setup_type.capitalize()} yolo gripper demo!")
    return True

def get_file_path(setup_type: Literal["video", "photo", "webcam"]) -> str:
    """Gets a file path. Will exit if the file path is not found.

    Returns:
        str: The file path to read from, or an empty string if `setup_type` is a webcam
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
            sys.exit("Unsupported file type!")
    file_path: str = find_file(os.getcwd(), file_name)
    
    # Ensure that the file at file_path exists
    if not os.path.isfile(file_path):
        print(f"Absolute path: {file_path}")
        sys.exit(f"Could not find file: {file_path}")
    
    assert file_path # Ensure that file_path is not Unknown
    assert check_file_extension_valid(file_path=file_path, setup_type=setup_type)
    return file_path

def ask_user_if_overwrite() -> bool:
    """Asks the user if they want to overwrite a file.

    Returns:
        bool: The user's response
    """
    print(f"Warning: file already exists.")
    _YES_RESPONSES: Final[list[str]] = ["y", "yes"]
    _NO_RESPONSES: Final[list[str]] = ["n", "no"]
    while True:
        res: str = input("Overwrite file? (y/n)").strip().lower()
        if not res:
            res = "y"
        if res in _YES_RESPONSES:
            return True
        if res in _NO_RESPONSES:
            return False
        print("Invalid input. Please try again.")

def get_destination_path(setup_type: Literal["video", "photo", "webcam"]) -> str:
    """Gets a file path.

    Returns:
        str: The file path to create.
    """
    
    _default_file_name: str = "example_annotated.jpg" if setup_type == "photo" else "example_annotated.mp4" if setup_type == "video" else "webcam_annotated.mp4" if setup_type == "webcam" else "UNKNOWN SETUP TYPE"
    file_path: str
    while True:
        file_name: str = input(f"Enter the name of the file to export to (default is {_default_file_name}): ").strip()
        
        if not file_name:
            print("No file name provided, going with default.")
            if setup_type in ["video", "photo", "webcam"]:
                file_name = _default_file_name
            else:
                sys.exit("Unsupported file type!")
        file_path = find_file(os.getcwd(), file_name)

        # Ensure that the file at file_path exists
        if os.path.isfile(file_path):
            print(f"Absolute path: {file_path}")
            if not ask_user_if_overwrite():
                continue
        if not file_path:
            file_path = os.path.join(os.getcwd(), file_name)
        break
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
        yolo.export(destination_path='example_annotated.mp4')

        >>> # Specify file and export photo
        yolo = YoloGripperDetections(setup_type='photo')
        yolo.export(source_path='example.jpg', destination_path='example_annotated.jpg')
    """
    MODEL_NAME: Final[str] = 'yolo11s-seg.pt'
    # 0 is person
    # fork 42, knife 43, mouse 64, cell phone 67, scissors 76, teddy bear 77, toothbrush 79
    TRACKED_CLASSES: Final[list[int]] = [42, 43, 64, 67, 76, 77, 79]

    font: int = cv2.FONT_HERSHEY_SIMPLEX
    setup_yolo(MODEL_NAME) # Download YOLO if it doesn't exist
    model: Model = YOLO(MODEL_NAME)

    def __init__(self, setup_type: Literal['webcam', 'video', 'photo'] | None = None) -> None:
        self.setup_type: Literal['webcam', 'video', 'photo'] = setup_type or get_setup_type()

        # Ensure that setup_type is a valid type
        if self.setup_type not in ['webcam', 'video', 'photo']:
            self.setup_type = get_setup_type()

        self.camera_source: int = 0
    
    def __call__(self, path: str | None = None) -> None:
        """Calls `display()`.

        Args:
            path (str, optional): The path to the photo/video file, or `None` to prompt the user. Use `None` to indicate webcam as well. Defaults to `None`.
        """
        self.display(path=path)

    @staticmethod
    def add_detected_to_frame(frame: cv2.typing.MatLike, detected_classes: list[str]) -> cv2.typing.MatLike:
        class_text: str = ', '.join(detected_classes)
        cv2.putText(frame, text=f"Detected: {class_text}", org=(10, 70), fontFace=YoloGripperDetection.font, fontScale=0.5, color=(0, 0, 255), thickness=1)
        return frame
    
    @staticmethod
    def clamp_frame(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
        height, width = frame.shape[:2]
        _MAX_WIDTH: Final[int] = 640
        _MAX_HEIGHT: Final[int] = 640
        _ratio = min(_MAX_WIDTH / width, _MAX_HEIGHT / height )

        _width, _height = int(width * _ratio), int(height * _ratio)

        _frame = cv2.resize(frame, (_width, _height), interpolation=cv2.INTER_AREA)
        
        # Make a white background canvas
        white_canvas = np.full((_MAX_WIDTH, _MAX_HEIGHT, 3), (255, 255, 255), dtype=np.uint8)
        
        # Calculate the center of the image
        x_offset = (_MAX_WIDTH - _width) // 2
        y_offset = (_MAX_HEIGHT - _height) // 2
        
        # Place the resized image on the canvas
        white_canvas[y_offset:y_offset + _height, x_offset:x_offset + _width] = _frame
        
        return white_canvas
    
    def detect(self, source_frame: cv2.typing.MatLike) -> tuple[list[Results], list[str]]:
        """The core of YoloGripperDetection. Will run the main algorithm on the provided `source_frame` and return the YOLO results and the detected classes.

        Args:
            source_frame (cv2.typing.MatLike): The frame to run the algorithm on.

        Returns:
            tuple (list[Results], list[str]): The results and detected_classes of `source_frame`.
        """
        results: list[Results] = YoloGripperDetection.model.track(source_frame, classes=YoloGripperDetection.TRACKED_CLASSES)
        hill_climb(results, source_frame, verbose=True)
        
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
        frame: cv2.typing.MatLike = self.clamp_frame(frame) # Clamp image to a viewable size
        results, detected_classes = self.detect(frame)
        
        return frame, detected_classes
    

    def video_detection(self, path: str | None = None) -> Generator[tuple[cv2.typing.MatLike, list[str]], None, None]:
        """Detects results on a `cv2.VideoCapture`.

        Args:
            path (str | None, optional): The path to the video file, or `None` to prompt the user. Use `None` to indicate webcam as well. Defaults to `None`.

        Yields:
            Generator (tuple[cv2.typing.MatLike, list[str]], None, None): A Generator with the results of each frame. See the example for how to use the generator.

        Example:
            ```
            for frame, detected_classes in self.video_detection(path):
                cv2.imshow("Gripper", frame)
            cv2.destroyAllWindows()
            ```
        """
        # Ensure only those with a cap are availible at this point
        assert self.setup_type == 'webcam' or self.setup_type == 'video'
        assert (self.setup_type == 'video' and path is not None) or (self.setup_type == 'webcam' and (path is None or path == ""))

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
            self.add_detected_to_frame(frame=frame, detected_classes=detected_classes)
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

        cap.release() # Close the VideoCapture

    def _display_photo(self, path: str) -> NoReturn:
        """Displays the photo detected from `photo_detection`.

        Args:
            path (str, optional): The path to the photo to display. Will ask user for input if it is `None`. Defaults to `None`.

        Returns:
            NoReturn: Does not return anything. Will exit the program.
        """
        assert self.setup_type == "photo"

        frame, detected_classes = self.photo_detection(path)
        # frame = self.clamp_frame(frame) # Doesn't no much but it sort of helps

        frame = self.add_detected_to_frame(frame=frame, detected_classes=detected_classes)
        # Add information to quit to frame
        cv2.putText(frame, text="Press any key to quit", org=(0, frame.shape[0] - 10), fontFace=YoloGripperDetection.font, fontScale=0.5, color=(0, 0, 255))

        # Display the annotated frame
        cv2.imshow("YOLO Inference", frame)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        cv2.destroyAllWindows()
        exit(1)
    
    def _display_video(self, path: str | None = None) -> None:
        """Displays the video to the screen.

        Args:
            path (str | None, optional): The path to the video file, or `None` to prompt the user. Use `None` to indicate webcam as well. Defaults to `None`.
        """
        for frame, detected_classes in self.video_detection(path):
            # Display the keybind information
            # Note that this logic is separate from the actual key handling. This is because it needs to change cap.
            # Not good for maintainability.
            cv2.putText(frame, text="Press q to quit", org=(0, frame.shape[0] - 10), fontFace=YoloGripperDetection.font, fontScale=0.5, color=(0, 0, 255))

            if self.setup_type == 'webcam':
                cv2.putText(frame, text="Press c to switch webcam source", org=(0, frame.shape[0] - 30), fontFace=YoloGripperDetection.font, fontScale=0.5, color=(0, 0, 255))
                
            frame: cv2.typing.MatLike = self.add_detected_to_frame(frame=frame, detected_classes=detected_classes)
            cv2.imshow("Gripper", frame)
        cv2.destroyAllWindows()

    def display(self, path: str | None = None) -> None:
        """Displays the webcam/file located in `path`.

        Args:
            path (str, optional): The path to the photo/video to display. Will ask user for input if it is `None`. Defaults to `None`.
        """

        _path: str = path or get_file_path(self.setup_type)
        if self.setup_type not in ["webcam"]:
            # Ensure that the user isn't passing an invalid file type
            # e.g. a jpg for a video
            check_file_extension_valid(_path, self.setup_type)
        if self.setup_type == "photo":
            self._display_photo(_path)
        if self.setup_type == "video":
            self._display_video(_path)
        if self.setup_type == "webcam":
            self._display_video()
    
    def _export_photo(self, source_path: str, destination_path: str) -> None:
        """Exports the photo from `source_path` to `destination_path`. Will ask for user input if `destination_path` is `None`.

        Args:
            source_path (str | None, optional): The path to the file to read. `None` will ask user for input. Defauts to `None`.
            destination_path (str | None, optional): The destination to put the exported photo. Defaults to `source_path`_annotated.jpg.
        """
        frame, detected_classes = self.photo_detection(source_path)
        # frame = self.clamp_frame(frame)
        frame: cv2.typing.MatLike = self.add_detected_to_frame(frame=frame, detected_classes=detected_classes)
        cv2.imwrite(filename=destination_path, img=frame)
        print(f"Exported photo to {os.path.abspath(destination_path)}")
    
    def _export_video(self, source_path: str, destination_path: str) -> None:
        """Exports the video from `source_path` (or the webcam if `self.setup_type` is webcam) to `destination_path`.

        Args:
            source_path (str | None): The path to the file to read. `None` will ask user for input (unless `self.setup_type` is webcam). `None` if `self.setup_type` is webcam.
            destination_path (str | None): The destination to put the exported video. Defaults to `source_path`_annotated.mp4. If `self.setup_type` is webcam and `destination_path` is None, will default to `webcam_annotated.mp4`.
        """
        if self.setup_type == "webcam" and not destination_path:
            destination_path = "webcam_annotated.mp4"
        
        if self.setup_type == "video":
            print("Exporting video...")
        fourcc: int = cv2.VideoWriter.fourcc(*"mp4v")
        writer: cv2.VideoWriter | None = None

        # Get the FPS of the video
        _fps: int = 5

        if self.setup_type == "webcam":
            cap = cv2.VideoCapture(self.camera_source)
        else:
            cap = cv2.VideoCapture(source_path)

        if cap.isOpened():
            _fps = int(cap.get(cv2.CAP_PROP_FPS)) or 5  # Use default if unavailable
            cap.release()
        # TODO: For webcam, record the video, and then use detect on the exported video.
        # BUG: It is currently a REALLY bad FPS (at least on my machine)
        try:
            for frame, detected_classes in self.video_detection(source_path):
                if writer is None:
                    height, width = frame.shape[:2]
                    # Writer definition is here to dynamically get the width and height of the video
                    writer = cv2.VideoWriter(filename=destination_path, fourcc=fourcc, fps=_fps, frameSize=(width, height))
                
                frame = self.add_detected_to_frame(frame=frame, detected_classes=detected_classes)
                writer.write(frame)
                if self.setup_type == "webcam":
                    cv2.imshow("Webcam YOLO Gripper Detection", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
        finally:
            if writer:
                writer.release()
            cv2.destroyAllWindows()

    def export(self, source_path: str | None = None, destination_path: str | None = None) -> None:
        """Exports the photo/video from `source_path` (or the webcam if `self.setup_type` is webcam) to `destination_path`.

        Args:
            source_path (str | None): The path to the file to read. `None` will ask user for input (unless `self.setup_type` is webcam). `None` if `self.setup_type` is webcam.
            destination_path (str | None): The destination to put the exported photo/video. Defaults to `source_path`_annotated.jpg/mp4. If `self.setup_type` is webcam and `destination_path` is None, will default to `webcam_annotated.mp4`.
        """
        _PHOTO_TYPES: Final[list[str]] = ["photo"]
        _VIDEO_TYPES: Final[list[str]] = ["video", "webcam"]
        _IGNORE_SOURCE_PATH_TYPES: Final[list[str]] = ["webcam"]

        cleaned_source_path: str = source_path or ""
        if source_path is None and self.setup_type not in _IGNORE_SOURCE_PATH_TYPES:
            cleaned_source_path = get_file_path(setup_type=self.setup_type)

        cleaned_destination_path: str = destination_path or ""
        if destination_path is None:
            cleaned_destination_path = get_destination_path(setup_type=self.setup_type)

        if self.setup_type in _PHOTO_TYPES:
            self._export_photo(source_path=cleaned_source_path, destination_path=cleaned_destination_path)
        elif self.setup_type in _VIDEO_TYPES:
            self._export_video(source_path=cleaned_source_path, destination_path=cleaned_destination_path)
        else:
            sys.exit(f"Unknown setup_type: {self.setup_type}!")

if __name__ == "__main__":
    yolo = YoloGripperDetection("video")
    yolo.export()