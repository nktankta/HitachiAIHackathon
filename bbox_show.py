import cv2
import numpy as np
from bbox_detect import decode_image_by_join,mask_to_bboxes
from openvino.inference_engine import IECore,IENetwork

model_xml='open_model_zoo-master/tools/downloader/intel/text-detection-0003/FP32/text-detection-0003.xml'
model_bin='open_model_zoo-master/tools/downloader/intel/text-detection-0003/FP32/text-detection-0003.bin'
ie = IECore()

net = ie.read_network(model=model_xml, weights=model_bin)
ex_net = ie.load_network(net,"CPU")

img = cv2.imread("img2.jpg")
dataptr = net.input_info["Placeholder"].input_data

n, c, h, w = dataptr.shape


in_frame = cv2.resize(img, (w, h))
input_frame = in_frame.transpose((2, 0, 1)).reshape((n, c, h, w))

out = ex_net.infer({"Placeholder":input_frame})
linkage ,cls= out.values()

out = decode_image_by_join(cls[0,1],linkage[0].transpose(1,2,0),pixel_conf_threshold=0.4,link_conf_threshold=0.6)
bbox = mask_to_bboxes(out,out.shape,min_area=100,min_height=5)

bbox = [i.reshape(4,2)*4 for i in bbox]
in_frame = cv2.polylines(in_frame,bbox,True,(0,0,255))
cv2.imshow("window",in_frame)
cv2.waitKey(0)

