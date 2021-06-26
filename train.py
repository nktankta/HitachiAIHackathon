import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional
from torch import optim
from torch.utils.data import DataLoader
from dataloader import DateDataset
from ddrnet import get_seg_model
from tqdm import tqdm
import matplotlib.pyplot as plt
import kornia as K
import warnings
from segformer_pytorch import Segformer

warnings.simplefilter("once")

class MyAugmentation(nn.Module):
    def __init__(self):
        super(MyAugmentation, self).__init__()
        self.k2 = K.augmentation.RandomAffine(360, [0., 0.2], [0.8, 1.5], [0., 0.20])
        self.lap = K.filters.Laplacian(3)

    def forward(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        img_out = self.lap(self.k2(img))
        mask_out = self.k2(mask, self.k2._params)

        return img_out, mask_out

aug = MyAugmentation()

def train(net, dataloader, optimizer,  device="cuda"):
    net.train()  # Set network to train mode
    loss_sum = 0
    for i, (sr,label) in enumerate(tqdm(dataloader)):
        if sr.size(0)==1:
            continue
        optimizer.zero_grad()  # 勾配の初期化
        sr = sr.to(device)
        label = label.to(device)
        sr,label = aug(sr,label)
        out = net(sr)  # 予測
        weight =10 * label +  (1-label)
        loss = F.binary_cross_entropy_with_logits(out, label,pos_weight=weight)  # Get avarage loss of batch
        loss.backward()  # 勾配の計算
        loss_sum += loss.item()
        optimizer.step()  # パラメータの更新
        del sr, label, loss, out
    return loss_sum/(i+1)

def test(net, dataloader, device="cuda",show_img=True):
    net.eval()  # Set network to train mode
    loss_sum = 0
    for i, (sr,label) in enumerate(dataloader):
        sr = sr.to(device)
        label = label.to(device)
        out = net(K.filters.laplacian(sr,3))  # 予測
        weight = 10 * label +  (1 - label)
        loss = F.binary_cross_entropy_with_logits(out, label, pos_weight=weight)  # Get avarage loss of batch
        loss_sum += loss.item()
        if show_img:
            plt.imshow(torchvision.transforms.functional.to_pil_image(sr[0]))
            plt.show()
            plt.imshow(torchvision.transforms.functional.to_pil_image(torch.sigmoid(out[0])))
            plt.show()

        del sr, label, loss, out
    return loss_sum/(i+1)

def main(**kwargs):
    option = {
        "lr": 1e-4,
        "patience": 20,
        "batch_size": 4,
        "epochs": 101,
        "loss": "bce",
        "augment": "holflip_90_270",
        "exclude_noslide": True,
        "dilated": False,
        "no_pretrain": False,
        "dropblock": 0.0,
    }

    torch.backends.cudnn.benchmark=True
    net = Segformer(channels=1,num_classes=1,dims=(12,24,36,48),heads=(1,2,3,4))
    net.cuda()

    transform = transforms.Compose([
            transforms.ToTensor()
        ])
    dataset = DateDataset(transform=transform,Train=True,size_rate=4)
    dataloader = DataLoader(dataset,batch_size=option["batch_size"],shuffle=True)

    test_dataset = DateDataset(transform=transform,Train=False,size_rate=4)
    test_dataloader = DataLoader(test_dataset,batch_size=option["batch_size"],shuffle=True)

    optimizer = optim.Adam(net.parameters(), lr=option["lr"],weight_decay=1e-2)

    for epoch in range(1, option["epochs"] + 1):
        print("epoch :",epoch)
        # train
        loss = train(net, dataloader, optimizer)
        print(loss)
        val_loss = test(net,test_dataloader,show_img=epoch%10==0)
        print(val_loss)
        torch.save(net.state_dict(), f"./checkpoint_seg/ckpt_{epoch}_val_{val_loss:.4f}.pth")

if __name__ == '__main__':
    main()