import numpy as np
import matplotlib.pyplot as plt
import cv2



# In the blackfin, values range from [0, 255] while the formula is based on 
# values in the range [0,1]. Therefore we divide by 255 after the channels.
# This leaves the factor 100 with which all coefficients were multiplied.

def get_FD_CV(x, y, im):

    Cb = round(im[ y , x , 1 ])
    Cr = round(im[ y , x , 2 ])
    
    FD_CV = (1975 * Cb - 446 * Cr) / 255 - 818

    return FD_CV


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


imbgr = cv2.imread('imag.jpg')
imrgb = cv2.cvtColor(imbgr, cv2.COLOR_BGR2RGB)
print(imrgb.shape, imrgb.dtype)
print(imrgb.max(), imrgb.min())
show(imbgr)
show(imrgb)

im_ycrcb = cv2.cvtColor(imrgb, cv2.COLOR_BGR2YCrCb)
print(im_ycrcb.shape, im_ycrcb.dtype)
print(im_ycrcb.max(), im_ycrcb.min())
show(im_ycrcb)
print(im_ycrcb)
show(rgb2ycbcr(imrgb))

im_rgb = cv2.cvtColor(im_ycrcb, cv2.COLOR_YCrCb2BGR)
print(im_rgb.shape, im_rgb.dtype)
print(im_rgb.max(), im_rgb.min())
show(im_rgb)


print(get_FD_CV(im_ycrcb, 0, 0))