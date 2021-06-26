import cv2
from image_to_product import image2product
import torch
from ddrnet import get_seg_model
import numpy as np
import kornia as K
import ttach as tta
from square_detect import findSquares

file_path = "best_lap_model.pth"

net = get_seg_model(pretrained=False)
net.load_state_dict(torch.load(file_path))
net.eval()
transforms = tta.Compose([tta.Rotate90(angles=[0,90, 180,270])])
net= tta.SegmentationTTAWrapper(net, transforms, merge_mode='mean')
capture = cv2.VideoCapture(0)

while(True):
    ret, frame = capture.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(frame_gray, (512, 384))
    image = np.pad(image, [(64, 64), (0, 0)])

    frame = cv2.resize(frame, (512, 384))
    frame = np.pad(frame, [(64, 64),(0,0), (0, 0)])
    with torch.no_grad():
        frame_tensor = torch.from_numpy(image.copy()).view(1,1,512,512)/255.0
        frame_tensor = K.laplacian(frame_tensor, 3)
        predict = torch.sigmoid(net(frame_tensor))
        pred_np = predict[0].detach().numpy()
    pred_np = (pred_np.transpose(1,2,0)*255/np.max(pred_np)).astype(np.uint8)
    pred_np = cv2.resize(pred_np, (512, 512))
    _, bw = cv2.threshold(pred_np, 100, 255, cv2.THRESH_BINARY)
    frame_detect = findSquares(bw, frame.copy(), cond_area=100)
    pred_np = cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR)
    stack = np.concatenate([frame,pred_np],axis=1)[128:-128,:,:]
    stack2 = np.concatenate([frame_detect,cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)],axis=1)[128:-128,:,:]
    stack = np.concatenate([stack,stack2],axis=0)
    cv2.imshow('frame',stack)
    product = image2product(frame_gray)
    if product!="error":
        print(product)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()