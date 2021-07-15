import features 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import texture

#dataset_path = '../GroundTruth/LabelME/Images'
#dst_path = '../segmented/segmented_'
dataset_path = '../holidays_selection'
dst_path = '../segmented_holidays/segmented_'

def groundPixel(x, y):

    img_out[y, x, :] = np.array([0, 0, 0])
    
    return img

def getMaximumY(img):

    max_y = 0

    for x in range(img.shape[1]):
        for y in range(img.shape[0]):

            Y = img[y, x, 0]
            if Y > max_y: max_y = Y 

    return max_y

def segment_no_yco():

    imgYCC = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    height = img.shape[0]
    width = img.shape[1]
    patch_size = 10
    maxY = getMaximumY(img) #TODO: confirmar isto!

    relative_y = 0
    for y in range(height):
        for x in range(width):

            FD_YCV = features.get_FD_YCV(x, y, imgYCC)
            if FD_YCV <= 0.579766:
                Cr = imgYCC[y, x, 1]/255
                if Cr <= 0.587229:
                    if FD_YCV <= -0.769074:
                        groundPixel(x, y)
                    else:
                        patch_texture = texture.getPatchTexture(img, x, y, patch_size)
                        if patch_texture <= 0.006609:
                            Cr = imgYCC[y, x, 1]/255
                            if Cr > 0.493324:
                                groundPixel(x, y)
                        else:
                            groundPixel(x, y)
                else:
                    patch_texture = texture.getPatchTexture(img, x, y, patch_size)
                    if patch_texture > 0.01739:
                        groundPixel(x, y)
            else:
                patch_texture = texture.getPatchTexture(img, x, y, patch_size)
                if patch_texture <= 0.017807:
                    FD_CV = features.get_FD_CV(x, y, imgYCC)
                    if FD_CV <= -0.507478:
                        groundPixel(x, y)
                else:
                    if FD_YCV <= 2.120051:
                        if patch_texture <= 0.04282:
                            FD_CV = features.get_FD_CV(x, y, imgYCC)
                            if FD_CV <= -0.193133:
                                groundPixel(x, y)
                            else:
                                Y = imgYCC[x, y, 0] #TODO: confirmar isto!
                                if Y/maxY <= 0.66109:
                                    groundPixel(x, y)
                                    relative_y+=1
                        else:
                            groundPixel(x, y)
    
    print(relative_y)
                 
if __name__ == "__main__":

    # Test LabelME dataset
    '''for folder in os.listdir(dataset_path):
        path = os.path.join(dataset_path,folder)
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename))
            img_out = np.full(img.shape, 255)
            segment_no_yco()
            
            plt.imshow(img)
            plt.show()
            plt.imshow(img_out)
            plt.show()

            cv2.imwrite(dst_path+filename, img_out)'''

    # Test holidays dataset
    for filename in os.listdir(dataset_path):
        img = cv2.imread(os.path.join(dataset_path, filename))
        img_out = np.full(img.shape, 255)
        segment_no_yco()
        
<<<<<<< HEAD
        '''plt.imshow(img)
=======
#         para ver imagens lado a lado
#         img_out = np.uint8(img_out)

#         Hori = np.concatenate((img, img_out), axis=1)

#         cv2.imshow("sas",Hori)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

        
        plt.imshow(img)
>>>>>>> 92f4b7336f9a57f8e864d3f62332a0ec0265d900
        plt.show()
        plt.imshow(img_out)
        plt.show()'''

<<<<<<< HEAD
        cv2.imwrite(dst_path+filename, img_out)
=======
        cv2.imwrite("../segmented/segmented_"+filename, img_out)
>>>>>>> 92f4b7336f9a57f8e864d3f62332a0ec0265d900
