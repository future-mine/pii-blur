import cv2
import time
import numpy as np
# Load the cascade
face_cascade = cv2.CascadeClassifier('models/licence.xml')
# Read the input image
img = cv2.imread('test/Untitled1.jpg')

# scale_percent = 80 # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# # resize image
# resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# Detect faces
# faces = face_cascade.detectMultiScale(img, 1.1, 1)
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# tempImg = resized.copy()
# maskShape = (resized.shape[0], resized.shape[1], 1)
# mask = np.full(maskShape, 0, dtype=np.uint8)

# mask_inv = cv2.bitwise_not(mask)
# background_img = cv2.bitwise_and(resized, resized, mask = mask_inv)
# foreground_img = cv2.bitwise_and(tempImg, tempImg,mask = mask)
# dst = cv2.add(background_img, foreground_img)
# Display the output
scale_percent = 900 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
print(img.shape)
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
dst = cv2.GaussianBlur(resized,(9,9),cv2.BORDER_DEFAULT)
print(dst.shape)
cv2.imwrite('test.jpg', dst)
cv2.imshow('dst_rt', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()


