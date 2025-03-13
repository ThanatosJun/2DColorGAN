import torch
import config
from torchvision.utils import save_image

def save_some_examples(gen, val_loader, epoch, folder):
    img_sketch, img_reference, img_truth = next(iter(val_loader))
    img_sketch, img_reference, img_truth = img_sketch.to(config.DEVICE), img_reference.to(config.DEVICE), img_truth.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        img_fake = gen(img_sketch, img_reference)
        img_fake = img_fake * 0.5 + 0.5
        save_image(img_fake, folder + f"/gen_{epoch}.png")
        save_image(img_sketch * 0.5 + 0.5, folder + f"/sketch_{epoch}.png")
        save_image(img_reference * 0.5 + 0.5, folder + f"/reference_{epoch}.png")
        if epoch == 0:
            save_image(img_truth * 0.5 +0.5, folder + f"/truth_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="weight/my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading Checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    if "state_dice" in checkpoint:
        checkpoint = {
            "state_dict": checkpoint["state_dice"],
            "optimizer": checkpoint["optimizer"],
            # "learning_rate": checkpoint["learning_rate"],
        }
    # here I code wrong for "state_dice" before v4, I correct it for "state_dict" after v4  
    model.load_state_dict(checkpoint["state_dict"])
    # model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr