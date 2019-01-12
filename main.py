import cv2, os
from matplotlib import pyplot as plt
import numpy as np
import argparse
import time

frame_per_second = 6
miss_rate = 15
similar_rate = 80
path_bg = '/content/drive/My Drive/sc-duc/bg'
path = './data/vehicle'
# car_cascade = cv2.CascadeClassifier(os.path.join(path, 'cars.xml'))
ans = []

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

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#detect car => list of car [x,y,w,h]
def detectCar(imagePath):
    # img = cv2.imread(imagePath, 1)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # return car_cascade.detectMultiScale(gray, 1.1, 1)
    image = cv2.imread(imagePath)
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
    net.setInput(blob)
    # start = time.time()
    layerOutputs = net.forward(ln)
    # end = time.time()

    boxes = []
    # loop over each of the layer outputs
    for output in layerOutputs:
	    # loop over each of the detections
	    for detection in output:
		    # extract the class ID and confidence (i.e., probability) of
		    # the current object detection
		    scores = detection[5:]
		    classID = np.argmax(scores)
		    confidence = scores[classID]

		    # filter out weak predictions by ensuring the detected
		    # probability is greater than the minimum probability
		    if confidence > args["confidence"]:
			    # scale the bounding box coordinates back relative to the
			    # size of the image, keeping in mind that YOLO actually
			    # returns the center (x, y)-coordinates of the bounding
			    # box followed by the boxes' width and height
			    box = detection[0:4] * np.array([W, H, W, H])
			    (centerX, centerY, width, height) = box.astype("int")

			    # use the center (x, y)-coordinates to derive the top and
			    # and left corner of the bounding box
			    x = int(centerX - (width / 2))
			    y = int(centerY - (height / 2))

			    # update our list of bounding box coordinates, confidences,
			    # and class IDs
			    boxes.append([x, y, int(width), int(height)])
    return boxes                

#calculate intersection of 2 rectangle
def intersectArea(a, b):
    dx = min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])
    dy = min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    return 0

#calculate similarity (%)
def similar(a, b):
    # average = ((a[2]*a[3]) + (b[2]*b[3])) / 2
    minArea = min(a[2]*a[3], b[2]*b[3])
    intersect = intersectArea(a, b)
    return (float(intersect) / float(minArea)) * 100

#update list of car after 1 frame [x,y,w,h,t,tlast] (0->5)
def update(cars, oneFrame, time):
    for i in oneFrame:
        check = False
        co = 0
        #check if it exist before (%)
        for j in cars:
            if similar(i, j) > similar_rate:
                check = True
                j[5] = time
                j[2] = max(i[2]+i[0], j[2]+j[0]) - min(i[0], j[0])
                j[3] = max(i[3]+i[1], j[3]+j[1]) - min(i[1], j[1])
                j[0] = min(i[0], j[0])
                j[1] = min(i[1], j[1])
                
            co += 1
        #add car if it not exist before
        if check == False:
            newCar = [i[0], i[1], i[2], i[3], time, time]
            cars.append(newCar)
    #remove car don't exist anymore
    newList = []
    for i in cars:
        if time <= i[5] + miss_rate:
            newList.append(i)
        elif time - i[4] > 120*frame_per_second:
            ans.append(i)
    return newList

def compress():
    #print(ans)
    newList = []
    for x in ans:
        x[5] = x[4]
        if len(newList) == 0:
            newList.append(x)
        else:
            # while (len(newList) > 0) and (x[4] < newList[len(newList) - 1][5] + 3*frame_per_second):
            #     x[4] = min(x[4], newList[len(newList) - 1][4])
            #     newList.pop()
            # newList.append(x)
            check = 0
            for y in newList:
                # < 2' or same position => 1 anomaly 
                if (abs(x[5]-y[5]) < 120*frame_per_second) or (similar(x, y) > 50):
                    check = 1
                    y[5]=x[5]
                    break
            if check==0:
                newList.append(x)
    #print(newList)
    return newList

resultFinalPath = "result" + ".txt"
MainResult = open(os.path.join(path, resultFinalPath), "w")
#for each video
for i in range(1,101):
    path_oneVideo = os.path.join(path_bg, str(i))
    #if folder background exist
    if not os.path.exists(path_oneVideo):
        continue
    print("work with " + str(i))
    resultFinalPath = "result" + str(i) + ".txt"
    result = open(os.path.join(path, resultFinalPath), "w")
    cars = []
    ans = []
    imgs = os.listdir(path_oneVideo)
    imgs.sort()

    #in each image
    co = 0
    longest = 0
    for j in imgs:
        #result.write(os.path.join(path_oneVideo, j) + "\n")
        oneFrame = detectCar(os.path.join(path_oneVideo, j))
        cars = update(cars, oneFrame, co)

        # print for test 
        # result.write("\n")
        # result.write(str(co) + ": ")
        # for xx in cars:
        #     result.write("(" + str(xx[0]) + "," + str(xx[1]) + "," + str(xx[2]) + "," + str(xx[3]) + "," + str(xx[4]) + ") ")
        # result.write("\n")
        
        #check if one car exist > 120 second (first time)
        # for (x,y,w,h,t,tlast) in cars:
            #print(t)
            # if co- (120*frame_per_second) == t:
            #     result.write(str(t / (frame_per_second * 60 * 30))) #minute
            #     result.write(":")
            #     result.write(str(t / (frame_per_second * 30))) #second
            #     result.write(" ")
        co += 1
    
    for x in cars:
        if co - x[4] > 120*frame_per_second:
            ans.append(x)
    ans = compress()
    #print result
    for (x,y,w,h,t,tlast) in ans:
        # minute = t / (frame_per_second * 60)
        # second = (t % (frame_per_second * 60)) / frame_per_second
        # result.write(str(minute)) #minute
        # result.write(":")
        # result.write(str(second)) #second
        # result.write(" ")
        second = t / frame_per_second
        if second > 2:
            resultString = str(i) + " " + str(second)
            result.write(resultString + "\n")
            MainResult.write(resultString + "\n")
    # result.write("\n")

    result.close()

MainResult.close()