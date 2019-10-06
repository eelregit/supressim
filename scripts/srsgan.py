import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist
from torch.utils.data import RandomSampler

from supressim.srsgan import models
from supressim.srsgan.datasets import BoxesDataset

sample_path = "/home1/06431/yueyingn/test/test-0929/supressim/saved_samples/"
model_path = "/home1/06431/yueyingn/test/test-0929/supressim/saved_models/"

os.makedirs(sample_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n-epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--n-discriminator-blocks", type=int, default=4, help="number of discriminator blocks. Reduce this if the training image is small to avoid convolving away everything.")
parser.add_argument("--hr-glob-path", type=str, default="/scratch1/06589/yinli/dmo-50MPC-fixvel/high-resl/set?/output/PART_004/*.npy", help="glob pattern for hires data")
parser.add_argument("--batch-size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight-decay", type=float, default=0.0002, help="adam: weight decay (similar to L2 regularization)")
parser.add_argument("--n-cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--sample-interval", type=int, default=100, help="interval between saving samples")
parser.add_argument("--checkpoint-interval", type=int, default=-1, help="interval between model checkpoints")

args = parser.parse_args()

def setup():   
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

    # initialize the process group, 
    # init_method='env://' to use global variable set by torch.distributed.launch
    dist.init_process_group("gloo",init_method='env://') 

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)
    
def cleanup():
    dist.destroy_process_group()
    
    
def main():  
    
    setup()    
    rank = dist.get_rank()    
    world_size = dist.get_world_size()
    
    if rank ==0:
        print (args)
    
    print ("set up done, my rank = %d, world size = %d" % (rank, world_size))
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    generator = models.GeneratorResNet()
    discriminator = models.Discriminator()

    # Losses
    criterion_G = nn.MSELoss()
    criterion_D = nn.BCEWithLogitsLoss()
    #criterion_content = nn.L1Loss()

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    #feature_extractor = feature_extractor.to(device)
    
    pg1 = dist.new_group(range(dist.get_world_size()))
#     generator = DDP(generator,process_group=pg1,broadcast_buffers=False)
    generator = DDP(generator,broadcast_buffers=False)
    
    pg2 = dist.new_group(range(dist.get_world_size()))
#     discriminator = DDP(discriminator,process_group=pg2,broadcast_buffers=False)
    discriminator = DDP(discriminator,broadcast_buffers=False)
    
    criterion_G = criterion_G.to(device)
    criterion_D = criterion_D.to(device)
    #criterion_content = criterion_content.to(device)

    if args.epoch != 0:
        generator.load_state_dict(torch.load(model_path + "generator_%d.pth"))
        discriminator.load_state_dict(torch.load(model_path + "discriminator_%d.pth"))

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    mysampler = RandomSampler(range(0,5000))
    dataloader = DataLoader(
        BoxesDataset(args.hr_glob_path),
        batch_size=args.batch_size,
        sampler=mysampler,
        num_workers=args.n_cpu,
    )

    # ----------
    #  Training
    # ----------

    for epoch in range(args.epoch, args.n_epochs):
        for i, (lr_boxes, hr_boxes) in enumerate(dataloader):

            lr_boxes = lr_boxes.to(device)
            hr_boxes = hr_boxes.to(device)

            real = torch.ones(1, dtype=torch.float, device=device, requires_grad=False)  # broadcasting
            fake = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            sr_boxes = generator(lr_boxes)

            # Adversarial loss
            sr_guess = discriminator(sr_boxes)
            sr_guess, real, fake = torch.broadcast_tensors(sr_guess, real, fake)
            loss_G = criterion_G(sr_guess, real)

            ## Total loss
            #loss_G = loss_content + 1e-3 * loss_G

            loss_G.backward()
            # print (loss_G.grad.data)
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            hr_boxes = models.narrow_like(hr_boxes, sr_boxes)
            loss_real = criterion_D(discriminator(hr_boxes), real)
            loss_fake = criterion_D(discriminator(sr_boxes.detach()), fake)

            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            if rank == 0:
                sys.stdout.write(
                    "rank0: [Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
                    % (epoch, args.n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
                )
                sys.stdout.flush()
                
            batches = epoch * len(dataloader) + i
    
            if batches % args.sample_interval == 0:
            
            # log here to check that models on different ranks are synchronized while input sample are different            
                if rank == 0:                    
                    np.save(sample_path + "lr0_{}.npy".format(batches), lr_boxes.numpy())
                    np.save(sample_path + "hr0_{}.npy".format(batches), hr_boxes.numpy())
                    np.save(sample_path + "sr0_{}.npy".format(batches), sr_boxes.detach().numpy())
                    
                    torch.save(generator.state_dict(), model_path + "generator0_%d.pth" % batches)
                    torch.save(discriminator.state_dict(), model_path + "discriminator0_%d.pth" % batches)
                    
                if rank == 1:                    
                    np.save(sample_path + "lr1_{}.npy".format(batches), lr_boxes.numpy())
                    np.save(sample_path + "hr1_{}.npy".format(batches), hr_boxes.numpy())
                    np.save(sample_path + "sr1_{}.npy".format(batches), sr_boxes.detach().numpy())
                    
                    torch.save(generator.state_dict(), model_path + "generator1_%d.pth" % batches)
                    torch.save(discriminator.state_dict(), model_path + "discriminator1_%d.pth" % batches)
                                       

#         if args.checkpoint_interval != -1 and epoch % args.checkpoint_interval == 0:
#             if rank == 0:
#                 torch.save(generator.state_dict(), model_path + "generator_%d.pth" % epoch)
#                 torch.save(discriminator.state_dict(), model_path + "discriminator_%d.pth" % epoch)

    cleanup()


if __name__ == '__main__':
    
    main()

    
