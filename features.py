import cv2
import matplotlib.pyplot as plt
import numpy as np

def get_FD_CV(x, y, im):

    Cb = round(im[ y , x , 2 ])
    Cr = round(im[ y , x , 1 ])
    
    FD_CV = (19.75 * Cb - 4.46 * Cr) / 255 - 8.18 # TODO: Confirmar

    return FD_CV

def get_FD_YCV(x, y, imgYCC):

    Y = imgYCC[y, x, 0]
    Cb = imgYCC[y, x, 2] 
    Cr = imgYCC[y, x, 1]
     
    return (8.60*Y + 25.50*Cb - 5.01*Cr)/255 - 15.45 

def getPatchTexture(im, x, y, patch_size):
    
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

    center_pixel = im[y, x, 0]/2

    dx = -half_patch_size
    while dx <= half_patch_size:
        dy = -half_patch_size
        while dy <= half_patch_size:
            if not(dx == 0 and dy == 0):
                indx = x + dx
                indy = y + dy
                value = im[indy, indx, 0]/2
                texture += abs( int(value) - int(center_pixel) )
            dy += 1
        dx += 1
    
    texture /= (patch_size * patch_size - 1)

    return texture


def get_FD_RGB(x, y, imgRGB):

    R = imgRGB[y, x, 0]
    G = imgRGB[y, x, 1]
    B = imgRGB[y, x, 2]
    
    return ((-3.77*R - 1.25*G + 12.40*B)/255 - 4.62)

def get_FD_HSV(x, y, imgHSV):
    
    H = imgHSV[y, x, 0]
    S = imgHSV[y, x, 1]
    V = imgHSV[y, x, 2]
    
    return (3.35*H/179 + 2.55*S/255 + 8.58*V/255 - 7.51)

def get_yco(y, img):
    return (y/img.shape[0])


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
                

if __name__ == "__main__":

    img = cv2.imread('../imag.jpg')
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    height = imgYCC.shape[0]
    width = imgYCC.shape[1]
    patch_size = 10
    
    '''FD_YCC = get_FD_YCV(0, 0, imgYCC)
    print(FD_YCC)'''

    for y in range(0, height):
        for x in range(0, width):
            get_uniformity(x, y, imgYCC, patch_size)
    

