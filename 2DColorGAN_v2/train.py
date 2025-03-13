import os
import torch
import torch.nn as nn
import torch.optim as optim
import config
from generator_model import Generator
from discriminator_model import Discriminator
from dataset import AnimeDataset
from utils import save_some_examples, save_checkpoint, load_checkpoint
from early_stop import EarlyStopping
from loss import VGGPerceptualLoss, compute_emd
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_fun(
        disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, vgg_loss, d_scaler, g_scaler,
):
    loop = tqdm(loader, leave=True)
    for idx, (img_sketch, img_reference, img_truth)in enumerate(loop):
        img_sketch = img_sketch.to(config.DEVICE)
        img_reference = img_reference.to(config.DEVICE)
        img_truth = img_truth.to(config.DEVICE)
        # Train Discriminator
        for _ in range(1):
            with torch.cuda.amp.autocast():
                img_fake = gen(img_sketch, img_reference)
                D_real = disc(img_sketch, img_reference, img_truth)
                D_real_loss = bce(D_real, torch.ones_like(D_real))
                D_fake = disc(img_sketch, img_reference, img_fake.detach())
                D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2
            
            disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), max_norm=1.0)
            d_scaler.step(opt_disc)
            d_scaler.update()
        # Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(img_sketch, img_reference, img_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(img_fake, img_truth) * config.L1_LAMBDA
            VGG = vgg_loss(img_fake, img_truth) * config.VGG_LAMBDA
            emd = compute_emd(img_fake, img_truth) * config.EMD_LAMBDA
            G_loss = G_fake_loss + L1 + VGG +emd
            # G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=1.0)
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real = round(torch.sigmoid(D_real).mean().item(), 5),
                D_fake = round(torch.sigmoid(D_fake).mean().item(), 5),
                G_fake_loss = round(G_fake_loss.mean().item(), 5),
                L1 = round(L1.mean().item(), 5),
                VGG = round(VGG.mean().item(), 5),
                emd = round(emd.mean().item(), 5),
                D_loss = round(torch.sigmoid(D_loss).mean().item(), 5),
                G_loss = round(torch.sigmoid(G_loss).mean().item(), 5),
            )
            loop.refresh()
    return G_loss
def main():
    disc = Discriminator(in_channels=9).to(config.DEVICE)
    gen = Generator(in_channels=6, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    scheduler_disc = CosineAnnealingLR(opt_disc, T_max=0.75*config.NUM_EPOCHS)
    scheduler_gen = CosineAnnealingLR(opt_gen, T_max=0.75*config.NUM_EPOCHS)
    L1_loss = nn.L1Loss()
    BCE = nn.BCEWithLogitsLoss()
    VGG_loss = VGGPerceptualLoss()
    earstop = EarlyStopping(patience = 20, min_delta = 0.001)
    Min_loss = 10

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
    train_dataset = AnimeDataset(root_dir=config.TRAIN_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = AnimeDataset(root_dir=config.VAL_DIR)
    val_loader = DataLoader(
        val_dataset,
        batch_size=6,
        shuffle=False,
    )
    if not os.path.exists(config.EXAMPLE_DIR):
        os.makedirs(config.EXAMPLE_DIR)

    for epoch in range(config.NUM_EPOCHS):
        G_loss = train_fun(disc, gen, train_loader, opt_disc, opt_gen, L1_loss, BCE, VGG_loss, d_scaler, g_scaler)
        scheduler_disc.step()
        scheduler_gen.step()
        # val_loss = earstop.validate(gen, val_loader, L1_loss)
        # if config.SAVE_MODEL and G_loss < min_loss:
        if config.SAVE_MODEL and (G_loss < Min_loss or epoch % 5 == 0):
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        if epoch % 5 == 0:
            save_some_examples(gen, val_loader, epoch, folder=config.EXAMPLE_DIR)
        
        earstop(G_loss)
        print(f"G_loss = {G_loss}")
        if earstop.early_stop:
            print("Early stopping triggered. Stopping training.")
            break

if __name__ == "__main__":
    main()