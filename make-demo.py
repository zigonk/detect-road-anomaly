import cv2, os
from matplotlib import pyplot as plt
import numpy as np
import argparse
import time

path_bg = './data/all_imgs/bg'
path = './data/vehicle'
pathResult = './data/vehicle/demo_result'
car_cascade = cv2.CascadeClassifier(os.path.join(path, 'cars.xml'))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#detect car => list of car [x,y,w,h]
def detectCar(image):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # return car_cascade.detectMultiScale(gray, 1.1, 1)
    (H, W) = image.shape[:2]

path_oneVideo = os.path.join(path_bg, str(75))
imgs = os.listdir(path_oneVideo)
imgs.sort()

for j in imgs:
    img = cv2.imread(os.path.join(path_oneVideo, j), 1)
    cars = detectCar(img)
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    cv2.imwrite(os.path.join(pathResult, j), img)