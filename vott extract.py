import os
from PIL import Image,ImageDraw
import json
import cv2
import numpy as np
from scipy.spatial import distance

path = "./export/"
json_path = "テスト-export.json"

output_img_path = "./text_recognition/"
with open(os.path.join(path,json_path),"r",encoding="utf-8") as f:
    data = json.load(f)["assets"]

queues = []

for v in data.values():
    name = v["asset"]["name"]
    point = v["regions"]
    point = [i["points"] for i in point]
    if len(point)>0:
        queues.append([name,point])

print(queues)

def is_clockwise(arr):
    diagonal = arr[2]-arr[0]
    phi = np.arctan(diagonal[1]/(diagonal[0]+1e-5)) + (diagonal[0]<0) * np.pi
    middle = arr[1]-arr[0]
    phi2 = np.arctan(middle[1]/(middle[0]+1e-5))  + (middle[0]<0) * np.pi
    return (phi-phi2)% (2*np.pi) > np.pi

def get_norm(arr):
    arr_roll = np.roll(arr,1,0)
    delta = arr_roll -arr
    delta_norm = np.linalg.norm(delta,axis=1)
    return delta_norm


id = 0
for i,(file_path,squares) in enumerate(queues):
    img = np.array(Image.open(os.path.join(path,file_path)))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for square in squares:
        points = []
        for point in square:
            x = round(point["x"])
            y = round(point["y"])
            points.append((x,y))

        points = np.array(points,dtype=np.float32)
        if is_clockwise(points):
            points = np.flip(points,0)

        if points.shape!=(4,2):
            continue

        if False:
            in_frame = cv2.polylines(img, [points.astype(np.int32)], True, (0, 0, 255),thickness=5)
            cv2.imshow("window", cv2.resize(in_frame,(in_frame.shape[0]//5,in_frame.shape[1]//5)))
            cv2.waitKey(0)

        delta = get_norm(points)
        if delta[0] > delta[1]:
            points = np.roll(points,1,0)

        x,y = (200,80)
        arr = np.array([[0, 0], [x, 0], [x, y], [0, y]])

        for i in range(2):
            arr = np.roll(arr, 2, axis=0)
            M = cv2.getPerspectiveTransform(points, arr.astype(np.float32))
            i_trans = cv2.warpPerspective(img, M, tuple(np.max(arr, axis=0).astype(np.int32)))
            cv2.imwrite(os.path.join(output_img_path,f"{id:04}_{i}.jpg"),i_trans)
        id += 1