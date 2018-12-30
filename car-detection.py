import cv2, os
from matplotlib import pyplot as plt
path = './data/vehicle'
pathData = './data/vehicle/vehicle'
path_bg = './data/all_imgs/bg'
car_cascade = cv2.CascadeClassifier(os.path.join(path, 'cars.xml'))
img = cv2.imread(os.path.join(pathData, '00030.jpg'), 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect cars
cars = car_cascade.detectMultiScale(gray, 1.1, 1)

# Draw border
ncars = 0
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
    ncars = ncars + 1

# Show image

plt.figure(figsize=(10,20))
plt.imshow(img)
plt.show()