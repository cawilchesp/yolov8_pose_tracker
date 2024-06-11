from pathlib import Path

# Folder parameters
ROOT = Path('D:/Data')
SOURCE_FOLDER = ROOT
OUTPUT_FOLDER = ROOT
MODEL_FOLDER = ROOT / 'models' / 'yolov8'

# Source parameters
INPUT_VIDEO = 'office_demo.mp4'
OUTPUT_NAME = 'office_demo_yolov8_pose'

# Deep Learning model configuration
MODEL_WEIGHTS = 'yolov8x-pose.pt'

# Inference configuration
IMAGE_SIZE = 640
CONFIDENCE = 0.25
