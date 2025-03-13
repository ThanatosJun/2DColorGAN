import torch
import torch.nn as nn

class ColorHistogram(nn.Module):
    def __init__(self, bins=16):
        super(ColorHistogram, self).__init__()
        self.bins = bins

    def forward(self, img_reference):
        # 將像素值從 [-1, 1] 轉為 [0, 1]
        img_reference = (img_reference + 1) / 2  
        batch_size, channels, height, width = img_reference.shape
        histograms = []

        for c in range(channels):
            channel = img_reference[:, c, :, :].reshape(batch_size, -1)  # (B, H*W)
            # hist = torch.histc(channel, bins=self.bins, min=0.0, max=1.0)  # (B,)
            hist = torch.stack([torch.histc(channel[i], bins=self.bins, min=-1, max=1.0) for i in range(batch_size)], dim=0)  # (B, bins)
            hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-6)  # 防止除以零
            histograms.append(hist.unsqueeze(1))

        histograms = torch.cat(histograms, dim=1)  # (B, 3, bins)
        return histograms

class ColorEmbedding(nn.Module):
    def __init__(self, bins=16):
        super(ColorEmbedding, self).__init__()
        self.histogram = ColorHistogram(bins=bins)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # Output: (B, 64, 256, 256)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # Output: (B, 128, 128, 128)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # Output: (B, 256, 64, 64)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),  # Output: (B, 512, 32, 32)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)  # Output: (B, 512, 1, 1)
        )
        self.linear_feat = nn.Linear(512, 64)        # 將圖像特徵轉為 64 維
        self.linear_hist = nn.Linear(bins * 3, 64)   # 將直方圖轉為 64 維
        self.linear_out = nn.Linear(128, 3)          # 最終輸出 3 維顏色向量
        
    def forward(self, img_reference):
        hist = self.histogram(img_reference)  # (B, 3, bins)
        hist = hist.view(hist.size(0), -1)     # (B, 3*bins)
        hist_feat = self.linear_hist(hist)     # (B, 64)

        x = self.encoder(img_reference)       # (B, 256, 1, 1)
        x = x.view(x.size(0), -1)              # (B, 256)
        img_feat = self.linear_feat(x)         # (B, 64)

        combined = torch.cat([img_feat, hist_feat], dim=1)  # (B, 128)
        color_feat = self.linear_out(combined)              # (B, 3)
        color_feat = color_feat.unsqueeze(-1).unsqueeze(-1) # (B, 3, 1, 1)
        color_map = color_feat.expand(-1, -1, img_reference.shape[2], img_reference.shape[3])  # (B, 3, H, W)
        return color_map

def test():
    x = torch.randn((1, 3, 512, 512)) *2 -1
    colorEmbedding = ColorEmbedding()
    color_map = colorEmbedding(x)
    print(f"color_map = {color_map.shape}")

if __name__ == "__main__":
    test()