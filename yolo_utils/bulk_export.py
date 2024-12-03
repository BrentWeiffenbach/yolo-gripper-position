import os
from yolo_gripper_detection import YoloGripperDetection

def get_all_image_paths(directory: str) -> list[str]:
    for _, _, files in os.walk(directory):
        images: list[str] = []
        for file in files:
            if file.split(".")[-1] in ["bmp", "dib", "jpg", "jpeg", "jpe", "jp2", "png", "pbm", "pgm", "ppm", "sr", "ras", "tiff", "tif"]:
                images.append(file)
        return images
    return []

# Run using:
# python -m yolo_utils.bulk_export
root = "figures/figure_comparisons"
image_paths: list[str] = get_all_image_paths(directory=root)
yolo = YoloGripperDetection(setup_type='photo')
for image in image_paths:
    yolo.export(source_path=os.path.join(root, image), destination_path=os.path.join(os.path.join(root, "results"), image.split('.')[-2] + "_annotated." + image.split(".")[-1]))