import openvino
import cv2
import numpy as np
from bbox_detect import decode_image_by_join,mask_to_bboxes
from openvino.inference_engine import IECore,IENetwork

model_xml='open_model_zoo-master/tools/downloader/intel/text-detection-0003/FP32/text-detection-0003.xml'
model_bin='open_model_zoo-master/tools/downloader/intel/text-detection-0003/FP32/text-detection-0003.bin'
ie = IECore()

net = ie.read_network(model=model_xml, weights=model_bin)
ex_net = ie.load_network(net,"CPU")

img = cv2.imread("img.png")
dataptr = net.input_info["Placeholder"].input_data

n, c, h, w = dataptr.shape

in_frame = cv2.resize(img, (w, h))
input_frame = in_frame.transpose((2, 0, 1)).reshape((n, c, h, w))

out = ex_net.infer({"Placeholder":input_frame})
linkage ,cls= out.values()

out = decode_image_by_join(cls[0,1],linkage[0].transpose(1,2,0))
bbox = mask_to_bboxes(out,out.shape,min_area=10,min_height=10)

bbox = [i.reshape(4,2)*4 for i in bbox]
in_frame = cv2.polylines(in_frame,bbox,True,(0,0,255))
cv2.imshow("window",in_frame)
cv2.waitKey(0)

p_trans = []
for i in bbox:
    min_size = np.min(i,axis=0)
    max_size = np.max(i,axis=0)
    x,y = max_size-min_size
    ld_index = np.argmin(np.sum(np.abs(i-min_size),axis=1))
    arr = np.array([[0,0],[x,0],[x,y],[0,y]])
    arr = np.roll(arr,ld_index,axis=0)
    p_trans.append(arr)

for box,trans in zip(bbox,p_trans):
    M = cv2.getPerspectiveTransform(box.astype(np.float32),trans.astype(np.float32))
    print(type(in_frame))
    i_trans = cv2.warpPerspective(in_frame,M,np.max(trans,axis=0))
    cv2.imshow("window",i_trans)
    cv2.waitKey(0)