import features as feat
import config
import os
import cv2
import numpy as np
from time import time

dataset_path = config.dataset_path
dst_path = config.dst_path

def groundPixel(x, y):

    img_out[y, x, :] = np.array([0, 0, 0])
    
    return img

def segment_new_full_tree():
    
    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    height = img.shape[0]
    width = img.shape[1]
    patch_size = 10

    for y in range(height):
        for x in range(width):
            if feat.get_FD_RGB(x, y, img) <= 0.62251:
                if feat.get_yco(y, img) <= 0.45:
                    if feat.getPatchTexture(imgYCC, x, y, patch_size) <= 0.012883:
                        if feat.get_FD_HSV(x, y, img) <= -1.416186:
                            groundPixel(x, y)
                    else:
                        groundPixel(x, y)
                else:
                    groundPixel(x, y)
            else:
                if feat.get_yco(y, imgYCC) <= 0.533333:
                    if feat.get_uniformity(x, y, imgYCC, patch_size) <= 0.4976:
                        if feat.get_FD_RGB(x, y, img) <= 2.008314:
                            groundPixel(x, y)
                        else:
                            if feat.get_FD_RGB(x, y, img) <= 2.611804:
                                groundPixel(x, y)
                    else:
                        if feat.get_yco(y, img) <= 0.008333:
                            groundPixel(x, y)
                else:
                    groundPixel(x, y)

if __name__ == "__main__":

    start_time = time()
    small = (480, 640)
    res_img = np.zeros(small)
    
    # Test holidays dataset
    for filename in os.listdir(dataset_path):
        original_img = cv2.imread(os.path.join(dataset_path, filename))
        img = cv2.resize(original_img, (640, 480))
        img_out = np.full(img.shape, 255)
        segment_new_full_tree()
        
#         para ver imagens lado a lado
#         img_out = np.uint8(img_out)

#         Hori = np.concatenate((img, img_out), axis=1)

#         cv2.imshow("sas",Hori)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

        cv2.imwrite(dst_path+filename, img_out)

    '''for filename in os.listdir('../segmented3_holidays'):
            
        img = cv2.imread(dataset_path + filename)
        imgout = cv2.imread(os.path.join('../segmented3_holidays', filename))
        img_out = cv2.resize(imgout, (640, 480)) 

        Hori = np.concatenate((img, img_out), axis=1)

        cv2.imshow(filename,Hori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

    print("------ %s seconds ------" % (time() - start_time))