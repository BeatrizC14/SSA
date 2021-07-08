import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def getPatchTexture(im, x, y, patch_size):
    half_patch_size = patch_size // 2
    texture = 0

    imgWidth = len(im[0])

    x = half_patch_size if (x < half_patch_size) else x
    x = imgWidth - half_patch_size - 1 if (x >= imgWidth - half_patch_size) else x
    y = half_patch_size if (y < half_patch_size) else y
    y = imgWidth - half_patch_size - 1 if (y >= imgWidth - half_patch_size) else y

    ix = im.shape[1]//2
    iy = im.shape[0]//2

    center_pixel = im[iy, ix, 0]

    dx = -half_patch_size
    while dx <= half_patch_size:
        dy = -half_patch_size
        while dy <= half_patch_size:
            if not(dx == 0 and dy == 0):
                indx = x + dx
                indy = y + dy
                value = im[indy, indx, 0]
                texture += abs( int(value) - int(center_pixel) )
            dy += 1
        dx += 1
    
    texture /= (patch_size * patch_size - 1)

    return texture



def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return np.uint8(rgb.dot(xform.T))

def show(im):
    plt.imshow(im)
    plt.show()


imbgr = cv.imread('imag.jpg')
imrgb = cv.cvtColor(imbgr, cv.COLOR_BGR2RGB)

im_ycrcb = cv.cvtColor(imrgb, cv.COLOR_BGR2YCrCb)

im_rgb = cv.cvtColor(im_ycrcb, cv.COLOR_YCrCb2BGR)

print(getPatchTexture(im_ycrcb, 5, 5, 10))