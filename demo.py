import cv2
from image_to_product import image2product
import numpy as np
import pyrealsense2 as rs

pipeline = rs.pipeline()
pipeline.start()

while(True):
    frame = pipeline.wait_for_frames()
    fr = frame.get_color_frame().get_data()
    frame = np.array(fr)
    frame =  cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_show = cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
  #  cv2.imshow('frame',cv2.resize(frame,(frame.shape[0]//2,frame.shape[1]//2)))
    cv2.imshow('frame',frame_show)

    product = image2product(frame)
    if product!="error":
        print(product)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

