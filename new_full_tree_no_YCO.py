import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import config
import features as feat
from time import time

dataset_path = config.dataset_path
dst_path = config.dst_path

def groundPixel(x, y):

    img_out[y, x, 0] = 0
    img_out[y, x, 1] = 0
    img_out[y, x, 2] = 0
    return 

def getMaximumY(img):

    max_y = 0
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):

            Y = img[y, x, 0]
            if Y > max_y: max_y = Y 

    return max_y

def segment_new_full_tree_no_yco(img):

    imgYCC = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)

    for y in range(height):
        for x in range(width):
            FD_RGB = feat.get_FD_RGB(x, y, img)
            if FD_RGB <= 0.62251:
                FD_HSV = feat.get_FD_HSV(x, y, img)
                if FD_HSV <= -1.019178:
                    groundPixel(x, y)
                else:
                    PSD = feat.get_PSD(imgYCC, x, y, 10)/255
                    if PSD <= 0.012387:
                        grad = feat.get_gradient(img, x, y)
                        if grad <= 0.009187:
                            if PSD > 0.005984:
                                groundPixel(x, y)
                        else: 
                            groundPixel(x, y)
                    else:
                        groundPixel(x, y)
            else:
                uni = feat.get_uniformity(x, y, img, 10)
                if uni <= 0.4976:
                    if FD_RGB <= 2.143961:
                        groundPixel(x, y)
                    else:
                        FD_YCV = feat.get_FD_YCV(x, y, imgYCC)
                        if FD_YCV <= 2.697233:
                            groundPixel(x, y)
                else:
                    text = feat.getPatchTexture(img, x, y, 10)
                    FD_CV = feat.get_FD_CV(x, y, img)
                    if FD_CV <= -0.509495:
                        if text > 0.01055:
                            groundPixel(x, y)
                    else:
                        if text > 0.015701:
                            FD_YCV = feat.get_FD_YCV(x, y, imgYCC)
                            if FD_YCV <= 2.065021:
                                Y = imgYCC[y, x, 0]
                                maxY = getMaximumY(imgYCC)
                                if Y/maxY <= 0.85227:
                                    groundPixel(x, y)
                                else:
                                    if feat.get_grayness(img, x, y) <= 0.094118:
                                        groundPixel(x, y)
                

if __name__ == "__main__":

    times = []
    for filename in os.listdir(dataset_path):
    
        start = time()
        original_img = cv.imread(os.path.join(dataset_path, filename))
        img = cv.resize(original_img, (640, 480))
        img_out = np.full(img.shape, 255)
        height = img.shape[0]
        width = img.shape[1]

        segment_new_full_tree_no_yco(img)
        
        #cv.imwrite("../segmented_imag.jpg", img_out)
        cv.imwrite(dst_path+filename, img_out)

        times.append((time()-start))
        
    print("Time worst case: %f" % np.amax(times))
    print("Time best case: %f" % np.amin(times))

    
