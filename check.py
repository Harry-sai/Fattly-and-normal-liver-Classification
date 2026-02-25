import cv2
import numpy as np

mask = cv2.imread("data/predictions_filled/fatty_liver/1167.PNG", 0)
print(np.unique(mask))
print(np.sum(mask))
