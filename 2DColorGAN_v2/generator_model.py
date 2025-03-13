import torch
import torch.nn as nn
import config
from color_embedding import ColorEmbedding

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        # "super(Block, self).__init__()" is OK
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.3)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class Generator(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, features=64):
        super().__init__()
        self.sketch_weight = config.SKETCH_WEIGHT
        self.reference_weight = config.REFERENCE_WEIGHT
        self.colorEMB = ColorEmbedding()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features*2, act="leaky")
        self.down2 = Block(features*2, features*4, act="leaky")
        self.down3 = Block(features*4, features*8, act="leaky")
        self.down4 = Block(features*8, features*8, act="leaky")
        self.down5 = Block(features*8, features*8, act="leaky")
        self.down6 = Block(features*8, features*8, act="leaky")

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, 2, 1), 
            nn.ReLU(),
        )

        self.up1 = Block(features*8, features*8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up3 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up4 = Block(features*8*2, features*8, down=False, act="relu", use_dropout=True)
        self.up5 = Block(features*8*2, features*4, down=False, act="relu", use_dropout=True)
        self.up6 = Block(features*4*2, features*2, down=False, act="relu", use_dropout=True)
        self.up7 = Block(features*2*2, features, down=False, act="relu", use_dropout=True)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=features*2, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, img_sketch, img_reference): 
        img_reference = self.colorEMB(img_reference)
        x = torch.cat([img_sketch * self.sketch_weight, img_reference * self.reference_weight], dim=1)
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        # 確保特徵圖大小匹配
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, self._resize(d7, up1)], 1))
        up3 = self.up3(torch.cat([up2, self._resize(d6, up2)], 1))
        up4 = self.up4(torch.cat([up3, self._resize(d5, up3)], 1))
        up5 = self.up5(torch.cat([up4, self._resize(d4, up4)], 1))
        up6 = self.up6(torch.cat([up5, self._resize(d3, up5)], 1))
        up7 = self.up7(torch.cat([up6, self._resize(d2, up6)], 1))

        return self.final_up(torch.cat([up7, self._resize(d1, up7)], 1))
    
    def _resize(self, x, target):
        return nn.functional.interpolate(x, size=target.shape[2:], mode="nearest")
        # return nn.functional.interpolate(x, size=target.shape[2:], mode="bilinear", align_corners=False)
    
def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Generator(in_channels=6, features=64)
    preds = model(x, y)
    print(preds.shape)

if __name__ == "__main__":
    test()