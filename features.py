import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

from numpy.core.fromnumeric import mean



def getMaximumY(img):
    
    max_y = 0

    for x in range(img.shape[1]):
        for y in range(img.shape[0]):

            Y = img[y, x, 0]
            if Y > max_y: max_y = Y 

    return max_y

def get_FD_CV(x, y, im):

    Cb = im[ y , x , 2 ]
    Cr = im[ y , x , 1 ]
    
    FD_CV = (19.75 * Cb - 4.46 * Cr) / 255 - 8.18 # TODO: Confirmar

    return FD_CV

def get_FD_YCV(x, y, imgYCC):

    Y = imgYCC[y, x, 0]
    Cb = imgYCC[y, x, 2] 
    Cr = imgYCC[y, x, 1]
     
    return (8.60*Y + 25.50*Cb - 5.01*Cr)/255 - 15.45 


def get_FD_RGB(x, y, imgBGR):

    B = imgBGR[y, x, 0]
    G = imgBGR[y, x, 1]
    R = imgBGR[y, x, 2]
    
    return ((-3.77*R - 1.25*G + 12.40*B)/255 - 4.62)

def get_FD_HSV(x, y, imgBGR):

    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    H = imgHSV[y, x, 0]
    S = imgHSV[y, x, 1]
    V = imgHSV[y, x, 2]
    
    return (3.35*H/179 + 2.55*S/255 + 8.58*V/255 - 7.51)


def get_yco(y, img):
    return (y/img.shape[0])



def get_grayness(im, x, y):

    Cb = im[ y , x , 2 ]
    Cr = im[ y , x , 1 ]

    g = (Cb/255 - 0.5)**2 + (Cr/255 - 0.5)**2

    return g



def getPatchTexture(im, x, y, patch_size):
    
    imgYCC = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
    half_patch_size = patch_size // 2
    texture = 0

    imgWidth = im.shape[1]
    imgHeight = im.shape[0]

    if (x < half_patch_size): x = half_patch_size
    if (x >= imgWidth - half_patch_size): x = imgWidth - half_patch_size - 1  
    if (y < half_patch_size): y = half_patch_size  
    if (y >= imgHeight - half_patch_size): y = imgHeight - half_patch_size - 1 

    '''ix = im.shape[1]//2
    iy = im.shape[0]//2'''

    center_pixel = imgYCC[y, x, 0]/2

    dx = -half_patch_size
    while dx <= half_patch_size:
        dy = -half_patch_size
        while dy <= half_patch_size:
            if not(dx == 0 and dy == 0):
                indx = x + dx
                indy = y + dy
                value = imgYCC[indy, indx, 0]/2
                texture += abs( int(value) - int(center_pixel) )
            dy += 1
        dx += 1
    
    texture /= (patch_size * patch_size - 1)

    return texture


def patch_mean(im, x, y, patch_size):

    imgYCC = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
    half_patch_size = patch_size // 2
    imgWidth = im.shape[1]
    imgHeight = im.shape[0]

    mean = 0

    if (x < half_patch_size): x = half_patch_size
    if (x >= imgWidth - half_patch_size): x = imgWidth - half_patch_size - 1  
    if (y < half_patch_size): y = half_patch_size  
    if (y >= imgHeight - half_patch_size): y = imgHeight - half_patch_size - 1 

    dx = -half_patch_size
    while dx <= half_patch_size:
        dy = -half_patch_size
        while dy <= half_patch_size:
            indx = x + dx
            indy = y + dy
            value = imgYCC[indy, indx, 0]/2
            mean += int(value); 
            dy += 1
        dx += 1
    
    mean /= (patch_size * patch_size)

    return mean


def get_PSD(width, height, img):
    L=0
    sum=0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = mean(img.shape[1], img.shape[0], gray)

    for y in range(0, height):
        for x in range(0, width):
            sum += (gray[y,x]-m)**2
            L+=1
    
    s = math.sqrt(sum/L)

    return s


def get_uniformity(x, y, im, patch_size):
    half_patch_size = patch_size // 2
    imgWidth = im.shape[1]
    imgHeight = im.shape[0]

    if (x < half_patch_size): x = half_patch_size
    if (x >= imgWidth - half_patch_size): x = imgWidth - half_patch_size - 1  
    if (y < half_patch_size): y = half_patch_size  
    if (y >= imgHeight - half_patch_size): y = imgHeight - half_patch_size - 1

    sqr = (x - half_patch_size, x + half_patch_size, y - half_patch_size, y + half_patch_size)
    patch = im[sqr[2]:sqr[3], sqr[0]:sqr[1], 0]/255 
    
    bins = 10
    hist, _ = np.histogram(patch, bins, (0, 1))

    # visualize intensity histogram
    '''n, edges, _ = plt.hist(patch.flatten(), bins, (0, 1)) 
    plt.show()'''

    p = hist/(patch_size**2)
    u = np.sum(p**2)

    return u
                

def get_gradient(img, x, y):

    if (x >= 0  and  x < img.shape[1]  and  y >= 0  and  y < img.shape[0]):
        
        if x > 0:
            Y1 = img[ y, x-1, 0]/2;
        else:
            Y1 = img[ y, 0, 0]/2;

        if x < (img.shape[1] - 1):
            Y2 = img[ y, x+1, 0]/2;
        else:
            Y2 = img[ y, img.shape[1]-1, 0]/2;
        
        dx = abs(round(Y2) - round(Y1));

        if y > 0:
            Y1 = img[ y-1, x, 0]/2;
        else:
            Y1 = img[ 0, x, 0]/2;
        
        if y < (img.shape[0] -1):
            Y2 = img[ y+1, x, 0]/2;
        else:
            Y2 = img[ img.shape[0] -1, x, 0]/2;
        
        dy = abs(round(Y2) - round(Y1));

    grad = (dx + dy)/255;

    return grad;




if __name__ == "__main__":

    #img = cv2.imread('../imag.jpg')
    img = cv2.imread('imag.jpg')
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    height = imgYCC.shape[0]
    width = imgYCC.shape[1]
    patch_size = 10
    
    '''FD_YCC = get_FD_YCV(0, 0, imgYCC)
    print(FD_YCC)'''

    for y in range(0, height):
        for x in range(0, width):
            m=get_grayness(img, x, y)
            print(m)
    
    
