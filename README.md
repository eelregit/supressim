# Super-Resolution Simulation

**Superceded by [map2map](https://github.com/eelregit/map2map)

## Run

Install pytorch
```shell
conda install pytorch -c pytorch
```

Install the package in dev mode:
```
pip install -e .
```

Set MKL threading on Frontera CLX node
```shell
export OMP_NUM_THREADS=56
```


## Models

* SRSGAN = SRGAN - content loss function + ...
* TODO: GAN to CGAN
    - Add `torch.nn.functional.interpolate` to upsample lores in preproc.py
    - Dataset, model
* TODO: concatenate white noise to lores in generator input
* TODO: ICNR initialization
  - <https://arxiv.org/abs/1707.02937>
* TODO: batchnorm after activation?
* TODO: activation, swish?


### Refs and Repos

* SRGAN <https://arxiv.org/abs/1609.04802>
* SRCNN <https://arxiv.org/abs/1501.00092>
* sub-pixel conv <https://arxiv.org/abs/1609.05158>
  <https://github.com/pytorch/examples/tree/master/super_resolution>
* some review <https://arxiv.org/abs/1808.03344>
  <https://github.com/YapengTian/Single-Image-Super-Resolution>
* pix2pix <https://arxiv.org/abs/1611.07004>
* <https://arxiv.org/abs/1801.09710>
* <https://github.com/imatge-upc/3D-GAN-superresolution>
* <https://github.com/yiyang7/Super_Resolution_with_CNNs_and_GANs>
* <https://github.com/xinntao/BasicSR>
* <https://github.com/xinntao/ESRGAN>


## Data

* augmentation, 8 flips x 6 permutations = 48 operations, on displacement and
  correspondingly velocity
* dark matter only simulations, 5 realizations, 50Mpc/h, z=7
    - lores 440^3 particles
    - hires 880^3 particles


## Accuracy

The generated small scale modes should have the right statistical property.

TODO: The following metrics can be used for evaluation
* PDF of the new displacements and velocities
* halo and subhalo abundances
* P(k) in the last 2-fold in k
* cross bispectrum, compare
    - `< delta_lo(k1) delta_lo(k2) delta_hi(k3) >`
    - `< delta_lo(k1) delta_lo(k2) delta_sr(k3) >`
  where `k3` is in the range of the generated short modes


## Efficiency

Parallelization

TODO: DDP
TODO: MKL-DNN vs DP
* MKL-DNN
  - `export MKL_NUM_THREADS=$NUM_CORES OMP_NUM_THREADS=$NUM_CORES`
  - alternatively, `torch.set_num_threads(num_cores)`
  - `torch.utils.data.DataLoader(num_workers=0)`, increase this number helps
  - <https://github.com/pytorch/pytorch/issues/9873>
* DataParallel and DistributedDataParallel, and torch.distributed
  - DP tutorial <https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html#sphx-glr-beginner-blitz-data-parallel-tutorial-py>
  - DDP tutorial <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>
  - torch.distributed tutorial <https://pytorch.org/tutorials/intermediate/dist_tuto.html>
  - <https://pytorch.org/docs/stable/nn.html#dataparallel-layers-multi-gpu-distributed>
  - <https://pytorch.org/docs/stable/distributed.html#basics>
    DDP "differs from the kinds of parallelism provided by Multiprocessing package - torch.multiprocessing and torch.nn.DataParallel() in that it supports multiple network-connected machines and in that the user must explicitly launch a separate copy of the main training script for each process."
  - performance difference <https://discuss.pytorch.org/t/why-torch-nn-parallel-distributeddataparallel-runs-faster-than-torch-nn-dataparallel-on-single-machine-with-multi-gpu/32977>
* multiprocessing
  - <https://pytorch.org/docs/stable/multiprocessing.html>
  - <https://pytorch.org/docs/stable/notes/multiprocessing.html>
  - Hogwild, or asynchronous multiprocess training
  - MNIST Hogwild example <https://github.com/pytorch/examples/tree/master/mnist_hogwild>
  - notes favor DataParallel <https://pytorch.org/docs/stable/notes/cuda.html#cuda-nn-dataparallel-instead>
* Model Parallel
  - <https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html>
