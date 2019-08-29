import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import *
from datasets import BoxesDataset

sample_path = "saved_samples/"
model_path = "saved_models/"
os.makedirs(sample_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n-epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--hr-glob-path", type=str, default="/scratch1/06589/yinli/dmo-50MPC-fixvel/high-resl/set?/output/PART_004/*.npy", help="glob pattern for hires data")
parser.add_argument("--batch-size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay-epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n-cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample-interval", type=int, default=100, help="interval between saving samples")
parser.add_argument("--checkpoint-interval", type=int, default=-1, help="interval between model checkpoints")
args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

generator = GeneratorResNet()
discriminator = Discriminator()
#feature_extractor = FeatureExtractor()

## Set feature extractor to inference mode
#feature_extractor.eval()

# Losses
criterion_GAN = nn.MSELoss()  # FIXME this MSE loss is strange
#criterion_content = nn.L1Loss()

generator = generator.to(device)
discriminator = discriminator.to(device)
#feature_extractor = feature_extractor.to(device)
criterion_GAN = criterion_GAN.to(device)
#criterion_content = criterion_content.to(device)

if args.epoch != 0:
    generator.load_state_dict(torch.load(model_path + "generator_%d.pth"))
    discriminator.load_state_dict(torch.load(model_path + "discriminator_%d.pth"))

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

dataloader = DataLoader(
    BoxesDataset(args.hr_glob_path),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.n_cpu,
)

# ----------
#  Training
# ----------

for epoch in range(args.epoch, args.n_epochs):
    for i, (lr_boxes, hr_boxes) in enumerate(dataloader):

        lr_boxes = lr_boxes.to(device)
        hr_boxes = hr_boxes.to(device)

        yes = torch.ones(1, dtype=torch.float, device=device, requires_grad=False)  # broadcasting
        no = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        sr_boxes = generator(lr_boxes)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(sr_boxes), yes)

        ## Content loss
        #gen_features = feature_extractor(gen_hr)
        #real_features = feature_extractor(imgs_hr)
        #loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        #loss_G = loss_content + 1e-3 * loss_GAN
        loss_G = loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        hr_boxes = narrow_like(hr_boxes, sr_boxes)
        loss_real = criterion_GAN(discriminator(hr_boxes), yes)
        loss_fake = criterion_GAN(discriminator(sr_boxes.detach()), no)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
            % (epoch, args.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
        )
        sys.stdout.flush()

        batches = epoch * len(dataloader) + i
        if batches % args.sample_interval == 0:
            #lr_boxes = nn.functional.interpolate(lr_boxes, scale_factor=2)
            np.save(sample_path + "lr_{}.npy".format(batches), lr_boxes.numpy())
            np.save(sample_path + "hr_{}.npy".format(batches), hr_boxes.numpy())
            np.save(sample_path + "sr_{}.npy".format(batches), sr_boxes.detach().numpy())

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        torch.save(generator.state_dict(), model_path + "generator_%d.pth" % epoch)
        torch.save(discriminator.state_dict(), model_path + "discriminator_%d.pth" % epoch)
