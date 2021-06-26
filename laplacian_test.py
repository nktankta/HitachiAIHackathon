import cv2
import numpy as np


img_path = "dataset/valid/img/0010.png"
img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
dst = cv2.Laplacian(img, cv2.CV_64F,ksize=3)
cv2.imshow("window",dst)
cv2.waitKey(0)
