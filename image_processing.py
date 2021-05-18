import cv2
import os
import numpy as np
from include import config
from imutils import paths
from matplotlib import pyplot as plt


imagePaths = list(paths.list_images(config.ORIG_IMAGES))

clahe = cv2.createCLAHE(clipLimit=24, tileGridSize=(8,8))

for (i, imagePath) in enumerate(imagePaths):

    img = cv2.imread(imagePath,0)

    #cl1 = clahe.apply(img)

    blur = cv2.medianBlur(img, 1)
    dst = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)
    dst = cv2.medianBlur(dst, 9)

    filename = "cells_{}.png".format(i+1)

    outPath = os.path.sep.join([config.ORIG_IMAGES_THRESHOLD, filename])

    cv2.imwrite(outPath, dst)


#cv2.imshow('Second image', img)
"""
img_to_yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
img = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
"""
#img = cv2.equalizeHist(img)


"""
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.plot(hist, color='gray' )

plt.xlabel('intensidad de iluminacion')
plt.ylabel('cantidad de pixeles')
plt.show()

cv2.destroyAllWindows()

cv2.waitKey(0)
"""

"""
#print(img[0][0])

#cv2.imshow('Original image', img)
#cv2.waitKey(0)

#height, width = img.shape[0:2]

#contrast_img = cv2.addWeighted(img, 1.5, np.zeros(img.shape, img.dtype), 0, 0)
#
cv2.imshow('Original Image', img)
#
#cv2.imshow('Contrast Image', contrast_img)

#processed_image = cv2.medianBlur(img, 3)
#cv2.imshow('Median Filter Processing', processed_image)

kernel = np.ones((3,3),np.float32)/9
processed_image1 = cv2.filter2D(img,-1,kernel)
# display image
cv2.imshow('Mean Filter Processing', processed_image1)

#edge_img = cv2.Canny(img,5,50)

#cv2.imshow("Detected Edges", edge_img)

cv2.waitKey(0)

"""