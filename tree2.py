import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import texture

def groundPixel(x, y):

    img_out[y, x, 0] = 0
    img_out[y, x, 1] = 0
    img_out[y, x, 2] = 0
    return 

def segment_hsv(hsv_img):

    for y in range(height):
        for x in range(width):
            if hsv_img[y, x, 2] <= 0.635294:
                if hsv_img[y, x, 0] <= 0.542517:
                    groundPixel(x, y)
                else: 
                    if hsv_img[y, x, 2] <= 0.458824:
                        groundPixel(x, y)
                    else:
                        if hsv_img[y, x, 1] <= 0.407258:
                            groundPixel(x, y)
                        else:
                            if hsv_img[y, x, 1] >= 0.601266:
                                groundPixel(x, y)
            else:
                if hsv_img[y, x, 0] <= 0.409091:
                    if hsv_img[y, x, 1] > 0.004032:
                        groundPixel(x, y)
                else: 
                    if hsv_img[y, x, 1] <= 0.191111:
                        if hsv_img[y, x, 2] <= 0.843137:
                            if hsv_img[y, x, 1] <= 0.073913:
                                groundPixel(x, y)
                            else:
                                if hsv_img[y, x, 2] <= 0.701961:
                                    groundPixel(x, y)
                            
    return


if __name__ == "__main__":
    img = cv.imread('imag.jpg')
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    height = hsv_img.shape[0]
    width = hsv_img.shape[1]
    img_out = np.full(hsv_img.shape, 255)

    for y in range(height):
        for x in range(width):
            hsv_img[y, x, 0] = hsv_img[y, x, 0]/360
            hsv_img[y, x, 1] = hsv_img[y, x, 1]/100
            hsv_img[y, x, 2] = hsv_img[y, x, 2]/100


    segment_hsv(hsv_img)

    plt.imshow(img_out)
    plt.show()
