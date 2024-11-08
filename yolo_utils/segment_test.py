import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

# Load the YOLO model
model = YOLO("yolo11n-seg.pt")  # segmentation model
names = model.names

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Failed to capture image from webcam")
        break

    # Run YOLO inference
    results = model.predict(im0)
    annotator = Annotator(im0, line_width=2)

    if results and results[0].masks is not None and results[0].boxes is not None:
        clss = results[0].boxes.cls.tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            color = colors(int(cls), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color, label=names[int(cls)], txt_color=txt_color)

            # Convert the mask to integer points
            mask = np.array(mask, dtype=np.int32)

            # Draw points along each segment of the mask boundary
            for i in range(len(mask)):
                # Get start and end points of the segment
                start_point = mask[i]
                end_point = mask[(i + 1) % len(mask)]  # Wrap around to the first point

                # Calculate the distance between the start and end points
                distance = np.linalg.norm(start_point - end_point)
                
                # Ensure at least one point is created along the segment
                num_points = max(int(distance // 5), 1)

                # Interpolate points along the line segment
                for j in range(num_points + 1):
                    # Linear interpolation between start_point and end_point
                    point = start_point + j / num_points * (end_point - start_point)
                    cv2.circle(im0, (int(point[0]), int(point[1])), radius=3, color=(0, 0, 0), thickness=-1)

    # Display the annotated frame
    cv2.imshow("instance-segmentation", im0)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
