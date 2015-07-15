CIFAR-10 classification framework, run this command in terminal (Linux only):
- Install CUDA: 
apt-get install nvidia-331
apt-get install nvidia-cuda-toolkit

-Install Torch7 (https://github.com/torch/ezinstall):
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash

-Install packages:
luarocks install torch
luarocks install nn
luarocks install cutorch
luarocks install cunn
luarocks install image

- Download and transform data to Lua filetype (https://github.com/soumith/cifar.torch/blob/master/Cifar10BinToTensor.lua): 
th download_original_file.lua

- train + validate + test models (change the model you want to test following instructions in "validating.lua"): 
th validation.lua

* Development environment:
- Ubuntu 14.04
- Tesla K20 GPU
- 32 Gb RAM

* Note: some code might not run because of the following reasons:
- You do not have a proper GPU. 
- Your data and model do not compatible.