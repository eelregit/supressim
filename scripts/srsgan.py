import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from supressim.srsgan import models
from supressim.srsgan.datasets import BoxesDataset

sample_path = "saved_samples/"
model_path = "saved_models/"
os.makedirs(sample_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n-epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--n-generator-residual-blocks", type=int, default=2, help="number of generator residual blocks. Reduce this if the training image is small to avoid convolving away everything.")
parser.add_argument("--n-discriminator-blocks", type=int, default=2, help="number of discriminator blocks. Reduce this if the training image is small to avoid convolving away everything.")
parser.add_argument("--lambda-content", type=float, default=100, help="weight for content loss.")
parser.add_argument("--cgan", type=bool, default=True, help="use conditional GAN in the discriminator.")
parser.add_argument("--hr-glob-path", type=str, default="/scratch1/06589/yinli/dmo-50MPC-fixvel/high-resl/set?/output/PART_004/*.npy", help="glob pattern for hires data")
parser.add_argument("--batch-size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr-G", type=float, default=0.0001, help="learning rate for generator")
parser.add_argument("--lr-D", type=float, default=0.0004, help="learning rate for discriminator")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
#parser.add_argument("--weight-decay", type=float, default=0.0002, help="adam: weight decay (similar to L2 regularization)")
#parser.add_argument("--decay-epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n-cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample-interval", type=int, default=100, help="interval between saving samples")
parser.add_argument("--checkpoint-interval", type=int, default=-1, help="interval between model checkpoints")
args = parser.parse_args()
print(args)

generator = models.GeneratorResNet(n_residual_blocks=args.n_generator_residual_blocks)
discriminator = models.Discriminator(n_blocks=args.n_discriminator_blocks)
criterion_adversarial = nn.BCEWithLogitsLoss()
criterion_content = nn.SmoothL1Loss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)

if args.epoch != 0:
    generator.load_state_dict(torch.load(model_path + "generator_{}.pth".format(args.epoch)))
    discriminator.load_state_dict(torch.load(model_path + "discriminator_{}.pth".format(args.epoch)))

generator = generator.to(device)
discriminator = discriminator.to(device)
criterion_adversarial = criterion_adversarial.to(device)
criterion_content = criterion_content.to(device)

#optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
#optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2), weight_decay=args.weight_decay)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr_D, betas=(args.b1, args.b2))

dataloader = DataLoader(
    BoxesDataset(args.hr_glob_path, augment=False),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.n_cpu,
)

# ----------
#  Training
# ----------

torch.manual_seed(42 + args.epoch)  # FIXME

for epoch in range(args.epoch, args.n_epochs):
    for i, (lr_boxes, hr_boxes) in enumerate(dataloader):

        lr_boxes = lr_boxes.to(device)
        hr_boxes = hr_boxes.to(device)

        real = torch.ones(1, dtype=torch.float, device=device, requires_grad=False)  # broadcasting
        fake = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

        sr_boxes = generator(lr_boxes)

        hr_boxes = models.narrow_like(hr_boxes, sr_boxes)
        if args.cgan:
            lu_boxes = F.interpolate(lr_boxes, scale_factor=2, mode='nearest')
            lu_boxes = models.narrow_like(lu_boxes, sr_boxes)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        loss_G_content = criterion_content(sr_boxes, hr_boxes)

        if args.cgan:
            sr_boxes = torch.cat([lu_boxes, sr_boxes], dim=1)
            hr_boxes = torch.cat([lu_boxes, hr_boxes], dim=1)

        sr_guess = discriminator(sr_boxes)
        sr_guess, real, fake = torch.broadcast_tensors(sr_guess, real, fake)
        loss_G_adversarial = criterion_adversarial(sr_guess, real)

        loss_G = args.lambda_content * loss_G_content + loss_G_adversarial

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        loss_D_real = criterion_adversarial(discriminator(hr_boxes), real)
        loss_D_fake = criterion_adversarial(discriminator(sr_boxes.detach()), fake)

        loss_D = (loss_D_real + loss_D_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        sys.stdout.write("[Epoch {}/{}] [Batch {}/{}] [D loss: {}] "
            "[G loss: {}] [G content loss: {}] [G adversarial loss: {}]\n".format(
                epoch, args.n_epochs, i, len(dataloader), loss_D.item(),
                loss_G.item(), loss_G_content.item(), loss_G_adversarial.item()
            )
        )
        sys.stdout.flush()

        batches = epoch * len(dataloader) + i
        if batches % args.sample_interval == 0:
            if args.cgan:
                sr_boxes = sr_boxes[:, sr_boxes.shape[1] // 2 :]
                hr_boxes = hr_boxes[:, hr_boxes.shape[1] // 2 :]

            np.save(sample_path + "lr_{}.npy".format(batches), lr_boxes.cpu().numpy())
            np.save(sample_path + "hr_{}.npy".format(batches), hr_boxes.cpu().numpy())
            np.save(sample_path + "sr_{}.npy".format(batches), sr_boxes.detach().cpu().numpy())

    if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
        torch.save(generator.state_dict(), model_path + "generator_{}.pth".format(1 + epoch))
        torch.save(discriminator.state_dict(), model_path + "discriminator_{}.pth".format(1 + epoch))

    if torch.cuda.is_available():
        sys.stderr.write("max GPU mem allocated: {}\n".format(torch.cuda.max_memory_allocated()))
        sys.stderr.write("max GPU mem cached: {}\n".format(torch.cuda.max_memory_cached()))
