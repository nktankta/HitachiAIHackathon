import cv2
import numpy as np
from openvino.inference_engine import IECore,IENetwork
import matplotlib.pyplot as plt

model_xml='open_model_zoo-master/tools/downloader/intel/text-recognition-0012/FP32/text-recognition-0012.xml'
model_bin='open_model_zoo-master/tools/downloader/intel/text-recognition-0012/FP32/text-recognition-0012.bin'
ie = IECore()

net = ie.read_network(model=model_xml, weights=model_bin)
ex_net = ie.load_network(net,"CPU")

img = cv2.imread("text_recognition/0000_0.jpg",cv2.IMREAD_GRAYSCALE)

dataptr = net.input_info["Placeholder"].input_data

n, c, h, w = dataptr.shape

alpha = 0.8  # コントラスト項目
beta = 20  # 明るさ項目

# 明るさ・コントラスト操作
res_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

in_frame = cv2.resize(res_img, (w, h))
input_frame = in_frame.reshape((n, c, h, w))

plt.imshow(in_frame.squeeze())
plt.show()

out = ex_net.infer({"Placeholder":input_frame})
recog = list(out.values()).pop().squeeze()
recog = np.argmax(recog,axis=1)
print(recog)

chars = "0123456789abcdefghijklmnopqrstuvwxyz#"
ls = []
for i in chars:
    ls.append(i)
chars = np.array(ls)

string = ""
for i in chars[recog]:
    if i!="#":
        string += i
print(string)
