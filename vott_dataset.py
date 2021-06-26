import os
from PIL import Image,ImageDraw
import json

path = "./export3/"
json_path = "消費期限-export.json"

output_img_path = "./dataset/img"
output_label_path = "./dataset/label"

os.makedirs(output_img_path,exist_ok=True)
os.makedirs(output_label_path,exist_ok=True)

output_image_size = (512,512)
#output_label_size = (output_image_size[0]//4,output_image_size[1]//4)
output_label_size = (output_image_size[0],output_image_size[1])

with open(os.path.join(path,json_path),"r",encoding="utf-8") as f:
    data = json.load(f)["assets"]

queues = []
for v in data.values():
    name = v["asset"]["name"]
    point = v["regions"]
    point = [i["points"] for i in point]
    if len(point)>0:
        queues.append([name,point])
append = 74
print(queues)
for i,(file_path,squares) in enumerate(queues):
    img = Image.open(os.path.join(path,file_path))
    label = Image.new('RGB', img.size, (0, 0, 0))
    draw = ImageDraw.Draw(label)
    for square in squares:
        points = []
        for point in square:
            x = round(point["x"])
            y = round(point["y"])
            points.append((x,y))
        draw.polygon(points, (255,255,255))
    rate = img.size[0]/img.size[1]
    img_bg = Image.new('RGB', output_image_size, (0, 0, 0))
    label_bg = Image.new('RGB', output_label_size, (0, 0, 0))
    if rate>1:
        img = img.resize((output_image_size[0],int(output_image_size[1]/rate)))
        img_bg.paste(img, (0,(output_image_size[1]-img.size[1])//2))
        label = label.resize((output_label_size[0],int(output_label_size[1]/rate)))
        label_bg.paste(label, (0,(output_label_size[1]-label.size[1])//2))
    else:
        img = img.resize((int(output_image_size[0]*rate), output_image_size[1]))
        img_bg.paste(img, ((output_image_size[0]-img.size[0])//2,0))
        label = label.resize((int(output_label_size[0]*rate),output_label_size[1]))
        label_bg.paste(label, ((output_label_size[0]-label.size[0])//2,0))
    img_bg.save(os.path.join(output_img_path,f"{i+append:04}.png"))
    label_bg.save(os.path.join(output_label_path,f"{i+append:04}.png"))