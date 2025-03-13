import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import config
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:32]  # 取前 32 層
        self.vgg = self.vgg.to(config.DEVICE)
        for param in self.vgg.parameters():
            param.requires_grad = False  # 固定 VGG 權重

    def forward(self, x, y):
        #  將 [-1, 1] 範圍映射到 [0, 1]
        x = (x + 1) / 2  
        y = (y + 1) / 2  

        #  使用 ImageNet 標準化 (VGG 模型需求)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        y = TF.normalize(y, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        #  提取 VGG 特徵
        x_features = self.vgg(x.float())  
        y_features = self.vgg(y.float())  

        #  計算 L1 感知損失
        return F.l1_loss(x_features, y_features)
    
def compute_histogram_torch(image, bins=32):
    # 將像素範圍從 [0, 1] 映射到 [0, bins-1]
    image = (image * (bins - 1)).long()
    hist = []
    for c in range(image.shape[1]):
        channel = image[:, c, :, :].flatten()
        hist_c = torch.histc(channel.float(), bins=bins, min=0, max=bins-1)
        hist_c /= hist_c.sum()  # 正規化
        hist.append(hist_c)
    return torch.stack(hist, dim=0)  # (C, bins)

def compute_emd(gen_img, target_img, bins=32):
    # 計算生成圖像和目標圖像的直方圖
    gen_hist = compute_histogram_torch(gen_img, bins)
    target_hist = compute_histogram_torch(target_img, bins)
    
    # 計算累積分佈函數（CDF）
    cdf_gen = torch.cumsum(gen_hist, dim=1)
    cdf_target = torch.cumsum(target_hist, dim=1)
    
    # 計算 EMD
    emd = torch.mean(torch.abs(cdf_gen - cdf_target))
    return emd