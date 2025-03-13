import torch
import config
class EarlyStopping:
    """監測 loss 並在 loss 未改善時提早停止"""
    def __init__(self, patience=15, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    # def validate(self, gen, val_loader, l1_loss):
    #     """驗證模型，計算 L1 loss"""
    #     gen.eval()
    #     total_loss = 0
    #     with torch.no_grad():
    #         for img_sketch, img_reference, img_truth in val_loader:
    #             img_sketch = img_sketch.to(config.DEVICE)
    #             img_reference = img_reference.to(config.DEVICE)
    #             img_truth = img_truth.to(config.DEVICE)

    #             img_fake = gen(img_sketch, img_reference)
    #             loss = l1_loss(img_fake, img_truth)
    #             total_loss += loss.item()

    #     gen.train()
    #     return total_loss / len(val_loader)  # 回傳平均 loss