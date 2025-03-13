import numpy as np
import os
import config_test
from PIL import Image
from torch.utils.data import Dataset
class AnimeDatasetTest(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)
    
    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        image = image[:, :, :3]

        # Test Dataset combine two images, left is reference and right is sketch
        img_sketch = image[:, 512:, :]
        img_reference = image[:, :512, :]
        img_truth = img_reference.copy()
        augmentations = config_test.resized_transform(image=img_sketch, image0=img_reference, image1=img_truth)
        img_sketch = augmentations["image"]
        img_reference = augmentations["image0"]
        img_truth = augmentations["image1"]
        
        img_sketch = config_test.tensor_transform(image = img_sketch)["image"]
        img_reference = config_test.tensor_transform(image = img_reference)["image"]
        img_truth = config_test.tensor_transform(image = img_truth)["image"]

        return img_sketch, img_reference, img_truth

if __name__ == "__main__":
    dataset = AnimeDatasetTest(config_test.TEST_DIR)
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