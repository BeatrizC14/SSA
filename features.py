import cv2

def get_FD_YCV(x, y, imgYCC):

    Y = imgYCC[x, y, 0]
    Cr = imgYCC[x, y, 1]
    Cb = imgYCC[x, y, 2]  

    return 8.60*Y + 25.5*Cb - 5.1*Cr - 15.45  

def getPatchTexture(img, x, y, patch_size):
    return 0

if __name__ == "__main__":

    img = cv2.imread('../imag.jpg')
    imgYCC = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    
    FD_YCC = get_FD_YCV(0, 0, imgYCC)
    print(FD_YCC)

