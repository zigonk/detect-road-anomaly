import cv2, os
from matplotlib import pyplot as plt

frame_per_second = 6
miss_rate = 15
similar_rate = 40
path_bg = './data/all_imgs/bg'
path = './data/vehicle'
car_cascade = cv2.CascadeClassifier(os.path.join(path, 'cars.xml'))
ans = []

#detect car => list of car [x,y,w,h]
def detectCar(imagePath):
    img = cv2.imread(imagePath, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return car_cascade.detectMultiScale(gray, 1.1, 1)

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
    intersect = intersectArea(a, b)
    return (float(intersect) / float(average)) * 100

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
        if len(newList) == 0:
            newList.append(x)
        else:
            while (len(newList) > 0) and (x[4] < newList[len(newList) - 1][5] + 3*frame_per_second):
                x[4] = min(x[4], newList[len(newList) - 1][4])
                newList.pop()
            newList.append(x)
    #print(newList)
    return newList

result = open(os.path.join(path, "result.txt"), "w")
#for each video
for i in range(1,101):
    path_oneVideo = os.path.join(path_bg, str(i))
    #if folder background exist
    if not os.path.exists(path_oneVideo):
        continue
    print("work with " + str(i))
    result.write("Video " + str(i) + " : ")
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
        minute = t / (frame_per_second * 60)
        second = (t % (frame_per_second * 60)) / frame_per_second
        result.write(str(minute)) #minute
        result.write(":")
        result.write(str(second)) #second
        result.write(" ")
    result.write("\n")

result.close()
