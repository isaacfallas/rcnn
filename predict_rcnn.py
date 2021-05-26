# import the necessary packages
from include.nms import non_max_suppression
from include import config
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the our fine-tuned model and label binarizer from disk
print("[INFO] loading model and label binarizer...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())

# load the input image from disk
image = cv2.imread(args["image"])
image_grayscale = cv2.imread(args["image"],0)
#image = imutils.resize(image, width=500)

clone = image.copy()

print("[INFO] running selective search...")
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchQuality()
rects = ss.process()

proposals = []
boxes = []


for (x, y, w, h) in rects[:config.MAX_PROPOSALS_INFER]:
	# extract the region from the input image, convert it from BGR to
	# RGB channel ordering, and then resize it to the required input
	# dimensions of our trained CNN
	roi = image[y:y + h, x:x + w]
	roi = cv2.resize(roi, config.INPUT_DIMS)

	# further preprocess by the ROI
	roi = roi.astype("float") / 255.0
	#roi = roi.reshape((roi.shape[0], roi.shape[1], roi.shape[2]))
	
	# update our proposals and bounding boxes lists
	proposals.append(roi)
	boxes.append((x, y, x + w, y + h))


# convert the proposals and bounding boxes into NumPy arrays
proposals = np.array(proposals)
boxes = np.array(boxes, dtype="int32")
print("[INFO] proposal shape: {}".format(proposals.shape))

# classify each of the proposal ROIs using fine-tuned model
print("[INFO] classifying proposals...")
proba = model.predict(proposals)
#print(proba)

print("[INFO] applying NMS...")
labels = lb.classes_[np.argmax(proba, axis=1)]
#print(labels)
#print(type(labels))
idxs = np.where(labels == 'cell')[0]
#print(idxs)

boxes = boxes[idxs]
#print(boxes)
proba = proba[idxs][:, 0]
#print(proba)

idxs = np.where(proba >= config.MIN_PROBA)
boxes = boxes[idxs]
proba = proba[idxs]

#print(idxs)
#print(boxes)
#print(proba)

# loop over the bounding boxes and associated probabilities
for (box, prob) in zip(boxes, proba):
	# draw the bounding box, label, and probability on the image
	(startX, startY, endX, endY) = box
	cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 255, 255), 2)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	text= "Cell: {:.2f}%".format(prob * 100)
	#cv2.putText(clone, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

# show the output after *before* running NMS
# cv2.imshow("Before NMS", clone)
cv2.imwrite("Before_NMS.jpg", clone)

# run non-maxima suppression on the bounding boxes
boxIdxs = non_max_suppression(boxes, proba)

blur = cv2.medianBlur(image_grayscale, 1)
imageThrs = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 15)
imageThrs = cv2.medianBlur(imageThrs, 9)
imageThrs = cv2.cvtColor(imageThrs, cv2.COLOR_GRAY2BGR)

# loop over the bounding box indexes
for i in boxIdxs:
	# draw the bounding box, label, and probability on the image
	(startX, startY, endX, endY) = boxes[i]
	cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 255), 2)
	cv2.rectangle(imageThrs, (startX, startY), (endX, endY), (164, 73, 163), 2)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	text= "Cell: {:.2f}%".format(proba[i] * 100)
	#cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
	#cv2.putText(imageThrs, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (164, 73, 163), 2)

# show the output image *after* running NMS
# cv2.imshow("After NMS", image)
# cv2.imshow("Threshold", imageThrs)

cv2.imwrite("After_NMS.jpg", image)
cv2.imwrite("Threshold.jpg", imageThrs)

# cv2.waitKey(0)