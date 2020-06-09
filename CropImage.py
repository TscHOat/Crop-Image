import cv2
import numpy as np
import dlib
from matplotlib import pyplot as plt
import Util


clahe = cv2.createCLAHE(clipLimit=2.0)

# img = cv2.imread('external-content.duckduckgo.jpg')
img = cv2.imread('./KTP 10.jpeg')
# cv2.imshow("dada",img)
cv2.waitKey(0)
try:
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
except:
    print("Its not an image")
    quit()

gray = clahe.apply(gray)

detector = dlib.get_frontal_face_detector()

face = detector(gray)

if np.shape(img) == ():
    print("image error")
    quit()

# print(img)
if len(face)<1:
    print("No face")
    quit()
crop_image = img.copy()
x1 = face[0].left()
y1 = face[0].top()
x2 = face[0].right()
y2 = face[0].bottom()
img = img[int(y1-((y2-y1))):int(y2+((y2-y1))), int(x1-((x2-x1))):int(x2+(((x2-x1))))]
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# gray = cv2.equalizeHist(gray)

gray = clahe.apply(gray)
cv2.imshow("1",gray)

gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
ret2, th2 = cv2.threshold(gray_blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# th2 = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY , 115, 1)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# opening = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel,iterations=2)
close = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel, iterations=4)

cv2.imshow("2",close)

# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
kernel = np.ones((3,3), np.uint8)
close = cv2.erode(close,kernel,iterations=1)
cv2.imshow("3",close)


th = Util.auto_canny(close,.75)
kernel = np.ones((2,2), np.uint8)
th = cv2.dilate(th,kernel,iterations=2)

contours,hierarchy= cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 2)
cv2.imshow("4",img)

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(th, cmap='gray', interpolation='bicubic')
# plt.xticks([]),plt.yticks([])
# plt.show()
cv2.imshow("5",th2)
biggest = None
for x in contours:
    if biggest is None or cv2.contourArea(biggest) < cv2.contourArea(x):
          biggest = x


rect = cv2.minAreaRect(biggest)

rect = cv2.boxPoints(rect)
rect = np.int0(rect)

im_result = Util.fix_prespective(img,rect)

cv2.imshow('im_result', im_result)

cv2.waitKey(0)
