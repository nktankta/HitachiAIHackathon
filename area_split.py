import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "dataset/label/0000.png"
img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
retval,labels = cv2.connectedComponents(img)

plt.imshow(labels)
plt.show()