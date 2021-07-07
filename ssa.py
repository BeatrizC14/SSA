import features 
import numpy as np
import cv2

def groundPixel(x, y):

    img_out[x, y, :] = np.array([0, 0, 0])
    
    return img

def getMaximumY(img):

    max_y = 0

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):

            Y = img[x, y, 0]
            if Y > max_y: max_y = Y 

    return max_y

def segment_no_yco():

    imgYCC = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    height = img.shape[0]
    width = img.shape[1]
    patch_size = 10
    maxY = getMaximumY(img) #TODO: confirmar isto!

    for x in range(height):
        for y in range(width):

            FD_YCV = features.get_FD_YCV(x, y, imgYCC)
            if FD_YCV <= 0.579766:
                Cr = imgYCC[x, y, 1]
                if Cr <= 0.587229:
                    if FD_YCV <= -0.769074:
                        groundPixel(x, y)
                    else:
                        patch_texture = features.getPatchTexture(img, x, y, patch_size)
                        if patch_texture <= 0.006609:
                            Cr = imgYCC[x, y, 1]
                            if Cr > 0.493324:
                                groundPixel(x, y)
                        else:
                            groundPixel(x, y)
                else:
                    patch_texture = features.getPatchTexture(img, x, y, patch_size)
                    if patch_texture > 0.01739:
                        groundPixel(x, y)
            else:
                patch_texture = features.getPatchTexture(img, x, y, patch_size)
                if patch_texture <= 0.017807:
                    FD_YCV = features.get_FD_YCV(x, y, imgYCC)
                    if FD_YCV <= -0.507478:
                        groundPixel(x, y)
                else:
                    if FD_YCV <= 2.120051:
                        patch_texture = features.getPatchTexture(img, x, y, patch_size)
                        if patch_texture <= 0.04282:
                            FD_YCV = features.get_FD_YCV(x, y, imgYCC)
                            if FD_YCV <= -0.193133:
                                groundPixel(x, y)
                            else:
                                Y = imgYCC[x, y, 0] #TODO: confirmar isto!
                                if Y/maxY <= 0.66109:
                                    groundPixel(x, y)
                        else:
                            groundPixel(x, y)
                 
if __name__ == "__main__":

    img = cv2.imread('../imag.jpg')
    img_out = np.full(img.shape, 255)

    segment_no_yco()