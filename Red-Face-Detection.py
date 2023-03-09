# Yohanes Amadeo Marvell - 2301862260
# UAS Computer Vision

# import libraries
import cv2
import numpy as np 

# read image
img = cv2.imread('pic1.jpg')

# change image color from BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# lower limit of red
lower_red = np.array([161,155,84])

# higher limit of red
upper_red = np.array([179,255,255])

# create a mask from lower and higher limits and HSV image
red_mask = cv2.inRange(hsv, lower_red, upper_red)

# load the cascade classifier
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# detect faces from hsv image mask
faces = classifier.detectMultiScale(red_mask, 1.1, 4)

# draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# show image
cv2.imshow('img', img)
cv2.waitKey(0)