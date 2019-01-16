import cv2, os
from matplotlib import pyplot as plt
import numpy as np
import argparse
import time
import pprint as pp

accept_second = 110
end_accept_second = 80
frame_per_second = 6
miss_rate = 8*frame_per_second
confident_rate = 0.28
similar_rate = 50
path_bg = '/content/drive/My Drive/sc-duc/bg'
path = '/content/drive/My Drive/sc-duc/result'
# car_cascade = cv2.CascadeClassifier(os.path.join(path,'cars.xml'))
ans = []

from darkflow.net.build import TFNet

options = {"model": "cfg/yolo.cfg", 
           "load": "/content/drive/My Drive/sc-duc/model /yolo.weights", 
           "threshold": 0.1, 
           "gpu": 1.0}

tfnet = TFNet(options)

#detect car => list of car [x,y,w,h]
def detectCar(imagePath):
    carList = []
    original_img = cv2.imread(imagePath)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    results = tfnet.return_predict(original_img) 
    for result in results:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        confidence = result['confidence']
        label = result['label'] 
        check=0
        if (confidence > confident_rate) and (label=='person' or label=='bicycle' or label=='car' or label=='bus' or label=='train' or label=='truck' or label=='motorbike'):   
            check=1
        if check==1 and (btm_x-top_x)*(btm_y-top_y) < (410*800)/10:    
            carList.append((top_x, top_y, btm_x-top_x, btm_y-top_y, confidence))
    return carList         

#calculate intersection of 2 rectangle
def intersectArea(a, b):
    dx = min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])
    dy = min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
    return 0

#calculate similarity (%)
def similar(a, b):
    average = ((a[2]*a[3]) + (b[2]*b[3])) / 2
    # minArea = min(a[2]*a[3], b[2]*b[3])
    intersect = intersectArea(a, b)
    return (float(intersect) / float(average)) * 100

#update list of car after 1 frame [x,y,w,h,t,tlast] (0->5) 6 label
def update(cars, oneFrame, time):
    for i in oneFrame:
        check = False
        co = 0
        #check if it exist before (%)
        for j in cars:
            if similar(i, j) > similar_rate:
                check = True
                j[6] = i[4]
                j[5] = time
                j[2] = max(i[2]+i[0], j[2]+j[0]) - min(i[0], j[0])
                j[3] = max(i[3]+i[1], j[3]+j[1]) - min(i[1], j[1])
                j[0] = min(i[0], j[0])
                j[1] = min(i[1], j[1])
                
            co += 1
        #add car if it not exist before
        if check == False:
            newCar = [i[0], i[1], i[2], i[3], time, time, i[4]]
            cars.append(newCar)
    #remove car don't exist anymore
    newList = []
    for i in cars:
        if time <= i[5] + miss_rate:
            newList.append(i)
            # first time -> update ans
        elif time - i[4] > accept_second*frame_per_second:
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
                if (abs(x[5]-y[5]) < accept_second*frame_per_second) or (similar(x, y) > 30):
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
for i in range(89,101):
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
        if (cars != []) and co%(frame_per_second*60)==0:
            print(cars)
        co += 1
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
        
    
    for x in cars:
        if co - x[4] > end_accept_second*frame_per_second:
            ans.append(x)
    ans = compress()
    #print result
    for (x,y,w,h,t,tlast,label) in ans:
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
            print(resultString)
    # result.write("\n")

    result.close()

MainResult.close()