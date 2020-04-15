import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('../Lucas_Kanade_Template_Tracker/data-20200415T043230Z-001/data/car/frame*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()