import cv2
import torch
from ddrnet import get_seg_model
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import kornia as K
import glob
import ttach as tta

file_path = "best_lap_model.pth"

net = get_seg_model(pretrained=False)
net.load_state_dict(torch.load(file_path))
net.eval()
net.cuda()
transforms = tta.Compose([tta.Rotate90(angles=[0,90, 180,270])])
net= tta.SegmentationTTAWrapper(net, transforms, merge_mode='mean')

img_path = "dataset/valid/img/0011.png"
img_path = "test_image/0002.jpg"

for i in glob.glob("dataset/valid/img/*.png"):
    filename = os.path.basename(i)
    frame = cv2.imread(i)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    with torch.no_grad():
        frame_tensor = torch.from_numpy(image).cuda().view(1, 1,512,512) / 255.0
        frame_tensor = K.laplacian(frame_tensor,3)
        predict = torch.sigmoid(net(frame_tensor))
        pred_np = predict[0].detach().numpy()
    pred_np = (pred_np*255).astype(np.uint8)
    cv2.imwrite(os.path.join("refine_dataset/valid/img/", filename), pred_np[0])