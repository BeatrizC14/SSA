import cv2

def get_FD_CV(x, y, im):

    Cr = round(im[ y , x , 1 ])
    Cb = round(im[ y , x , 2 ])
    
    FD_CV = ((19.75 * Cb - 4.46 * Cr) / 255 - 8.18)*100 # TODO: Confirmar

    return FD_CV

def get_FD_YCV(x, y, imgYCC):

    Y = imgYCC[y, x, 0]
    Cr = imgYCC[y, x, 1]
    Cb = imgYCC[y, x, 2]  

    return ((8.60*Y + 25.50*Cb - 5.01*Cr)/255 - 15.45)*100  

def getPatchTexture(img, x, y, patch_size):
    return 0

if __name__ == "__main__":

    img = cv2.imread('../imag.jpg')
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    
    FD_YCC = get_FD_YCV(0, 0, imgYCC)
    print(FD_YCC)

