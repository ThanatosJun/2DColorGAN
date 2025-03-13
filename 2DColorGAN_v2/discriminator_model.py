import torch
import torch.nn as nn
import torch.nn.functional as F
from color_embedding import ColorEmbedding
from torch.nn.utils import spectral_norm

class CNNBLOCK(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode="reflect")
        )
        self.norm = nn.InstanceNorm2d(out_channels)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)  # 加入 Dropout 防止過擬合

    def forward(self, x):
        return self.dropout(self.leaky_relu(self.norm(self.conv(x))))

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, kernel_size=1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=1))
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma = nn.Parameter(torch.Tensor(0.01))

    def forward(self, x):
        batch, C, H, W = x.size()

        # 計算 Query、Key、Value
        q = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)  # (B, H*W, C//8)
        k = self.key(x).view(batch, -1, H * W)                     # (B, C//8, H*W)
        v = self.value(x).view(batch, -1, H * W)                   # (B, C, H*W)

        # 計算注意力權重並進行縮放
        scale = (q.size(-1) ** 0.5)
        attn = F.softmax((torch.bmm(q, k) / scale).clamp(-10, 10), dim=-1)
        # (B, H*W, H*W)

        # 計算輸出
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(batch, C, H, W)  # (B, C, H, W)

        # 輸出加上殘差連接，gamma 控制影響強度
        return self.gamma * out + x


class Discriminator(nn.Module):
    def __init__(self, in_channels=9, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBLOCK(in_channels, feature, stride=1 if feature == features[-1] else 2)
            )
            in_channels = feature

        # 加入 Self-Attention (提升全域感知能力)
        layers.append(SelfAttention(features[-1]))
        # layers.append(SelfAttention(features[-1]))

        # 最後一層輸出
        layers.append(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 4, 1, 1))
        )

        self.model = nn.Sequential(*layers)

        self.colorEMB = ColorEmbedding()

    def forward(self, img_sketch, img_reference, img_truth):
        img_reference = self.colorEMB(img_reference)
        x = torch.cat([img_sketch, img_reference, img_truth], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x
    
def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    z = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=9)
    preds = model(x, y, z)
    print(model)
    print(preds.shape)


if __name__ == "__main__":
    test()