# -*- coding: utf-8 -*-
import cv2, os
rt = '/content/drive/My Drive/track-dataset'
videos = os.listdir(rt)
#path for original frames
wrt_ori = './data/all_imgs/all'
#path for background frames
wrt_bg = '/content/drive/My Drive/sc-duc/bg'
# wrt_bg = './data/all_imgs/bg'
if not os.path.exists(wrt_ori):
    os.mkdir(wrt_ori)
if not os.path.exists(wrt_bg):
    os.mkdir(wrt_bg)

for video in videos:
    print (video)
    if not os.path.exists(os.path.join(wrt_bg, video.split('.')[0])):
        os.mkdir(os.path.join(wrt_bg, video.split('.')[0]))
    if not os.path.exists(os.path.join(wrt_ori, video.split('.')[0])):
        os.mkdir(os.path.join(wrt_ori, video.split('.')[0]))

    #read video
    cap = cv2.VideoCapture(os.path.join(rt, video))
    ret, frame = cap.read()

    #h, w, _ = frame.shape
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter(os.path.join(wrt, video), fourcc, 3, (w, h))

    #build MOG2 model
    bs = cv2.createBackgroundSubtractorMOG2(history=120)
    bs.setHistory(120)

    count = 1

    while ret:
        fg_mask = bs.apply(frame)
        bg_img = bs.getBackgroundImage()
        #out.write(bg_img)
        #filter frame
        if count%5 == 0 :
            cv2.imwrite(os.path.join(wrt_bg, video.split('.')[0], str(int(count)).zfill(5)+'.jpg'), bg_img)
            # cv2.imwrite(os.path.join(wrt_ori, video.split('.')[0], str(int(count)).zfill(5)+'.jpg'), frame)
        ret, frame = cap.read()
        count += 1
    #out.release()
    #quit()

