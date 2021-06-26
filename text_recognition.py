import cv2
import numpy as np
from openvino.inference_engine import IECore,IENetwork
import matplotlib.pyplot as plt
import glob
import datetime

class Recognizer():
    def __init__(self):
        model_xml='open_model_zoo-master/tools/downloader/intel/text-recognition-0012/FP32/text-recognition-0012.xml'
        model_bin='open_model_zoo-master/tools/downloader/intel/text-recognition-0012/FP32/text-recognition-0012.bin'
        ie = IECore()
        net = ie.read_network(model=model_xml, weights=model_bin)
        self.exe_net = ie.load_network(net,"CPU")

        dataptr = net.input_info["Placeholder"].input_data
        _,_,self.h, self.w = dataptr.shape

    def recognize(self,img=None,img_path=None):
        if img is None:
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

        alpha = 0.8  # コントラスト項目
        beta = 30  # 明るさ項目
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        img = cv2.GaussianBlur(img, (5,5), 0)
        img = cv2.dilate(img, (3,3), iterations=1)
        #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, (3,3))
        #mean = np.mean(img)*0.8
        #_,img =  cv2.threshold(img,int(mean),255,cv2.THRESH_BINARY)

        in_frame = cv2.resize(img, (self.w,self.h,))

        input_frame = in_frame.reshape((1,1, self.h,self.w))

        plt.imshow(input_frame.squeeze())
        plt.show()

        out = self.exe_net.infer({"Placeholder": input_frame})
        recog = list(out.values()).pop().squeeze()
        recog[:,10:-1] = -np.inf
        recog = np.argmax(recog, axis=1)
        chars = "0123456789abcdefghijklmnopqrstuvwxyz#"
        string = ""
        for i in recog:
            if i<10:
                string +=chars[i]
        return string

def refine(string):
    year = 0
    length = len(string)
    if length ==9:
        year = string[:2]
        month = string[4:-3]
        date = string[-2:]
    if length==8 or length==7:
        year = string[:2]
        month = string[3:-3]
        date = string[-2:]
    if length==5 or length==6:
        year = string[:2]
        month = string[2:-2]
        date = string[-2:]
    if year!=0:
        return f"{year}-{month}-{date}"
    return string

def test_date(string):
    try:
        dt = datetime.datetime.strptime(string, '%y-%m-%d')
        return dt > datetime.datetime(21,6,16)
    except:
        return False

if __name__ == '__main__':
    recog = Recognizer()
    count = 0
    for i in glob.glob("text_recognition/*.jpg"):
        rec = recog.recognize(img_path=i)
        if rec!="":
            refine_str = refine(rec)
            is_corr = test_date(refine_str)
            if is_corr:
                count +=1
                print(i,refine_str)
    print(count)