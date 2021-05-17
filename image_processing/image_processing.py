import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("cancer_cells.png",0)


clahe = cv2.createCLAHE(clipLimit=24, tileGridSize=(8,8))
cl1 = clahe.apply(img)

#t, dst = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)

blur = cv2.medianBlur(cl1, 1)
dst = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)
dst = cv2.medianBlur(dst, 9)

cv2.imwrite('clahe_img.jpg',cl1)

cv2.imwrite('threshold_img.jpg', dst)

#cv2.imshow('Original image', img)
#cv2.imshow('CLAHE', cl1)
#cv2.imshow('Blur', blur)
#cv2.imshow('Threshold', dst)

cv2.waitKey(0)









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