import torch
import os
import torch.optim as optim
import config_test
from generator_model import Generator
from utils import load_checkpoint
from dataset_test import AnimeDatasetTest
from torch.utils.data import DataLoader
from torchvision.utils import save_image

def test_and_save(gen, test_loader, folder):
    gen.eval()
    for idx, (img_sketch, img_reference, img_truth) in enumerate(test_loader):
        img_sketch, img_reference, img_truth = img_sketch.to(config_test.DEVICE), img_reference.to(config_test.DEVICE), img_truth.to(config_test.DEVICE)
        with torch.no_grad():
            img_fake = gen(img_sketch, img_reference)
            # save_image(img_fake * 0.5 + 0.5, folder + f"/gen_{idx}.png")
            # save_image(img_sketch * 0.5 + 0.5, folder + f"/sketch_{idx}.png")
            # save_image(img_reference * 0.5 + 0.5, folder + f"/reference_{idx}.png")
            # save_image(img_truth * 0.5 +0.5, folder + f"/truth_{idx}.png")
            img_combine = torch.cat([img_sketch, img_reference, img_fake],dim=2)
            save_image(img_combine * 0.5 + 0.5, folder + f"/combine_{idx}.png")
    gen.train()

def main():
    gen = Generator(in_channels=6, features=64).to(config_test.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config_test.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(config_test.CHECKPOINT_GEN_TEST, gen, opt_gen, config_test.LEARNING_RATE)

    test_dataset = AnimeDatasetTest(root_dir=config_test.TEST_DIR)
    test_loader = DataLoader(
        test_dataset,
        batch_size=3,
        shuffle=False,
    )
    if not os.path.exists(config_test.TEST_SAVE_DIR):
        os.makedirs(config_test.TEST_SAVE_DIR)
    test_and_save(gen, test_loader, config_test.TEST_SAVE_DIR)

if __name__ == "__main__":
    main()
    print(f"Finished GEN!")