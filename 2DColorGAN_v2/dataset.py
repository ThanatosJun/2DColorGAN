import numpy as np
import os
import config
from PIL import Image
from torch.utils.data import Dataset

class AnimeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)
        self.image_size = config.IMAGE_SIZE

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        img_sketch = image[:, :self.image_size, :]
        img_reference = image[:, self.image_size: 2 * self.image_size, :]
        img_truth = image[:, 2 * self.image_size:, :]

        img_sketch = img_sketch[:, :, :3]
        img_reference = img_reference[:, :, :3]
        img_truth = img_truth[:, :, :3]

        augmentations = config.resized_transform(image=img_sketch, image0=img_reference, image1=img_truth)
        img_sketch = augmentations["image"]
        img_reference = augmentations["image0"]
        img_truth = augmentations["image1"]

        img_sketch = config.sketch_transform(image=img_sketch)["image"]
        img_reference = config.reference_transform(image=img_reference)["image"]
        img_truth = config.truth_transform(image=img_truth)["image"]
        return img_sketch, img_reference, img_truth
    
if __name__ == "__main__":
    dataset = AnimeDataset("dataset/train/")
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    loader = DataLoader(dataset, batch_size=5)
    for x, y, z in loader:
        print(x.shape)
        print(y.shape)
        print(z.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()