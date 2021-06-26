import torch
from PIL import Image
import glob
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import kornia as K

class DateDataset(torch.utils.data.Dataset):
    def __init__(self,root="dataset/", transform = None,Train=True,size_rate=8):
        self.transform = transform
        self.img_path = f"{root}{'train' if Train else 'valid'}/img/"
        self.label_path = f"{root}{'train' if Train else 'valid'}/label/"
        self.size_rate = size_rate
        data = glob.glob(self.img_path+"*.png")
        self.data = [os.path.basename(i) for i in data]
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        file_name = self.data[idx]

        image = Image.open(os.path.join(self.img_path, file_name))
        label = Image.open(os.path.join(self.label_path, file_name))
        image = image.convert("L")
        label = label.convert("L")

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)




        if self.size_rate>1:
            h,w = label.size(1),label.size(2)
            label = label.view(1,1,h,w)
            label = F.interpolate(label,(h//self.size_rate,w//self.size_rate),mode="nearest")
            label = label.view(1,h//self.size_rate,w//self.size_rate)

        return image,label

class DateMargeDataset(torch.utils.data.Dataset):
    def __init__(self,root="dataset/", transform = None,Train=True,size_rate=8):
        self.transform = transform
        self.pred_path = f"{root}{'train' if Train else 'valid'}/img/"
        self.img_path = f"dataset/{'train' if Train else 'valid'}/img/"
        self.label_path = f"{root}{'train' if Train else 'valid'}/label/"
        self.size_rate = size_rate
        data = glob.glob(self.img_path+"*.png")
        self.data = [os.path.basename(i) for i in data]
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        file_name = self.data[idx]

        pred = Image.open(os.path.join(self.pred_path, file_name))
        image = Image.open(os.path.join(self.img_path, file_name))
        label = Image.open(os.path.join(self.label_path, file_name))
        pred = pred.convert("L")
        image = image.convert("L")
        image = image.resize((64,64))
        label = label.convert("L")

        if self.transform:
            pred = self.transform(pred)
            image = self.transform(image)
            label = self.transform(label)
        image = torch.cat([pred,image],axis=0)



        if self.size_rate>1:
            h,w = label.size(1),label.size(2)
            label = label.view(1,1,h,w)
            label = F.interpolate(label,(h//self.size_rate,w//self.size_rate),mode="nearest")
            label = label.view(1,h//self.size_rate,w//self.size_rate)

        return image,label


if __name__ == '__main__':
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    set = DateDataset(transform=transform)
    img,label = set[0]
    print(torch.max(label))