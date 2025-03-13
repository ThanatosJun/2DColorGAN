import torch
import albumentations as A
from albumentations import ToTensorV2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR = "dataset_test"
TEST_SAVE_DIR = "test25"
CHECKPOINT_GEN_TEST = "weight/2DColor_gen_v25.pth.tar"
LEARNING_RATE = 2e-4

resized_transform = A.Compose(
    [A.Resize(width=256, height=256)],
    additional_targets={"image0": "image","image1": "image"}
)

tensor_transform = A.Compose(
    [
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
    ToTensorV2(),
    ]
)