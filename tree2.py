import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import config

dataset_path = config.dataset_path
dst_path = config.dst_path

def groundPixel(x, y):

    img_out[y, x, 0] = 0
    img_out[y, x, 1] = 0
    img_out[y, x, 2] = 0
    return 

def segment_hsv(hsv_img):

    for y in range(height):
        for x in range(width):
            if hsv_img[y, x, 2]/255 <= 0.635294:
                if hsv_img[y, x, 0]/179 <= 0.542517:
                    groundPixel(x, y)
                else: 
                    if hsv_img[y, x, 2]/255 <= 0.458824:
                        groundPixel(x, y)
                    else:
                        if hsv_img[y, x, 1]/255 <= 0.407258:
                            groundPixel(x, y)
                        else:
                            if hsv_img[y, x, 1]/255 <= 0.601266:
                                groundPixel(x, y)
            else:
                if hsv_img[y, x, 0]/179 <= 0.409091:
                    if hsv_img[y, x, 1]/255 > 0.004032:
                        groundPixel(x, y)
                else: 
                    if hsv_img[y, x, 1]/255 <= 0.191111:
                        if hsv_img[y, x, 2]/255 <= 0.843137:
                            if hsv_img[y, x, 1]/255 <= 0.073913:
                                groundPixel(x, y)
                            else:
                                if hsv_img[y, x, 2]/255 <= 0.701961:
                                    groundPixel(x, y)
                            
    return


if __name__ == "__main__":
    '''img = cv.imread('../imag.jpg')
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)'''
    
    

    '''for y in range(height):
        for x in range(width):
            hsv_img[y, x, 0] = hsv_img[y, x, 0]/ 360
            hsv_img[y, x, 1] = hsv_img[y, x, 1]/ 100
            hsv_img[y, x, 2] = hsv_img[y, x, 2]/ 100

    segment_hsv(hsv_img)

    plt.imshow(img_out)
    plt.show()'''

    for filename in os.listdir(dataset_path):
        img = cv.imread(os.path.join(dataset_path, filename))
        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        img_out = np.full(img.shape, 255)
        height = hsv_img.shape[0]
        width = hsv_img.shape[1]

        segment_hsv(hsv_img)
        
        cv.imwrite(dst_path+filename, img_out)
