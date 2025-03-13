import torch
import albumentations as A
from albumentations import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset512x512/train"
VAL_DIR = "dataset512x512/val"
EXAMPLE_DIR = "example26"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 100
VGG_LAMBDA = 0.5
EMD_LAMBDA = 1
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "weight/2DColor_disc_v26.pth.tar"
CHECKPOINT_GEN = "weight/2DColor_gen_v26.pth.tar"
SKETCH_WEIGHT = 0.3
REFERENCE_WEIGHT = 0.7
# INPUT_SIZE = 256

resized_transform = A.Compose(
    [A.Resize(width=256, height=256)], additional_targets={"image0": "image","image1": "image"}
)

sketch_transform = A.Compose(
    [
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),  # 小幅模糊，模擬不同畫風
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

reference_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),  # 可保留
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

truth_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

# both_ske_tru_transform = A.Compose(
#     [
#         A.HorizontalFlip(p=0.5),
#     ]
# )