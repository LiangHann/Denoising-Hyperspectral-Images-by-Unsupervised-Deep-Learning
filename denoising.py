from __future__ import print_function
import matplotlib.pyplot as plt

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import scipy.io
import numpy as np
import cv2
import time
from models import *

import torch
import torch.optim

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = False
sigma = 25.5
sigma_ = sigma/255
s_vs_p = 0.5 # ratio between salt and pepper noise
amount = 0.1 # density of salt and pepper noise

## create path to save images
image_path_input_original = 'result/image/input_original'
image_path_input_noisy = 'result/image/input_noisy'
image_path_output = 'result/image/output'
if not os.path.isdir(image_path_input_original):
    os.mkdir(image_path_input_original)
if not os.path.isdir(image_path_input_noisy):
    os.mkdir(image_path_input_noisy)
if not os.path.isdir(image_path_output):
    os.mkdir(image_path_output)

## load denoising image and create the input noise tensor
file_name = 'data/denoising/denoising_DC_Mall_sigma_10.mat'
#file_name = 'data/denoising/denoising_PU_sigma_5.mat'
# ensure dimensions [0][1] are divisible by 32 (or 2^depth)!

mat = scipy.io.loadmat(file_name)
img_np = mat["y"]
img_noisy_np = mat["z"]
img_np = img_np.transpose(2,0,1)
img_var = torch.from_numpy(img_np).type(dtype)
img_noisy_np = img_noisy_np.transpose(2,0,1)
img_noisy_torch = torch.from_numpy(img_noisy_np).type(dtype)

## setup arguments
method = '2D'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 0.03 # 0 0.01 0.05 0.08
LR = 0.01

OPTIMIZER='adam' # 'LBFGS'
exp_weight=0.99

show_every = 200
save_every = 200

num_iter = 30000
input_depth = img_np.shape[0]  

## create DIP-Net
net = skip(input_depth, img_np.shape[0],  
       num_channels_down =   [16, 128, 128, 128, 128], # [128]*5,
       num_channels_up =     [16, 128, 128, 128, 128], # [128]*5,
       num_channels_skip =   [4]*5,  
       filter_size_up = 3, filter_size_down = 3, filter_skip_size=1,
       upsample_mode='bilinear', # downsample_mode='avg',
       need1x1_up=False,
       need_sigmoid=False, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    
net_input = get_noise(input_depth, method, (img_np.shape[1], img_np.shape[2])).type(dtype).detach()

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

## Loss
mse = torch.nn.MSELoss().type(dtype)
l1_loss = torch.nn.L1Loss().type(dtype)

img_noisy_torch = img_noisy_torch[None, :].cuda()

## format the original clean and noisy input for saving images
img_np_print = np.clip(img_np*255,0,255).astype(np.uint8)
if img_np_print.shape[0] == 1:
    img_np_print = img_np_print[0]
else:
    img_np_print = img_np_print.transpose(1, 2, 0)

img_noisy_np_print = np.clip(img_noisy_np*255,0,255).astype(np.uint8)
if img_noisy_np_print.shape[0] == 1:
    img_noisy_np_print = img_noisy_np_print[0]
else:
    img_noisy_np_print = img_noisy_np_print.transpose(1, 2, 0)


## optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0, 0.9), amsgrad=False)

## optimizing
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psnr_noisy_last = 0

iteration = 0

tic = time.perf_counter()

while iteration <= num_iter:
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    net.zero_grad()
    net.train()

    out, basis = net(net_input)

    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            
#    MSE_loss = mse(out, img_noisy_torch)
    L1_loss = l1_loss(out, img_noisy_torch)
#    TV_loss = tv_loss(out)
    total_loss = L1_loss
    total_loss.backward()
    optimizer.step()

    out_np = out.detach().cpu().squeeze().numpy()
    out_avg_np = out_avg.detach().cpu().squeeze().numpy()
  
    psnr_noisy = compare_psnr(img_noisy_np.astype(np.float32), np.clip(out_np, 0, 1))
    psnr_gt    = compare_psnr(img_np.astype(np.float32), np.clip(out_np, 0, 1)) 
    psnr_gt_sm = compare_psnr(img_np.astype(np.float32), np.clip(out_avg_np, 0, 1)) 
    ssim_gt    = compare_ssim(img_np.astype(np.float32), np.clip(out_np, 0, 1)) 
    ssim_gt_sm = compare_ssim(img_np.astype(np.float32), np.clip(out_avg_np, 0, 1)) 
    
    if iteration % show_every == 0:
        out_np = np.clip(out_np, 0, 1)
        out_avg_np = np.clip(out_avg_np, 0, 1)
        out_np_print = np.clip(out_np*255,0,255).astype(np.uint8)
        if out_np_print.shape[0] == 1:
            out_np_print = out_np_print[0]
        else:
            out_np_print = out_np_print.transpose(1, 2, 0)
        for i in range(out_np_print.shape[2]):
            if i % 20 == 0:
                cv2.imwrite('{}/{}_{}_output.jpg'.format(image_path_output, iteration, i), out_np_print[:, :, i])

        print("===> Iteration: {} Loss: {:.4f} PSNR_noisy: {:.4f} PSNR_gt: {:.4f} PSNR_gt_sm: {:.4f} SSIM_gt: {:.4f} SSIM_gt_sm: {:.4f}".format(
                     iteration, total_loss.item(), psnr_noisy, psnr_gt, psnr_gt_sm, ssim_gt, ssim_gt_sm))
    
#    if  iteration % save_every == 0:
#        scipy.io.savemat("results/result_denoising_2D_it%05d.mat" % (i), {'pred':out_np.transpose(1,2,0),
#                                                                          'pred_avg':out_avg_np.transpose(1,2,0)})

    # Backtracking
    if iteration % show_every:
        if psnr_noisy - psnr_noisy_last < -5: 
            print('Falling back to previous checkpoint.')
            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psnr_noisy_last = psnr_noisy
            
    iteration += 1


## save final images
for i in range(img_np_print.shape[2]):
    cv2.imwrite('{}/{}_input_original.jpg'.format(image_path_input_original, i), img_np_print[:, :, i])
for i in range(img_noisy_np_print.shape[2]):
    cv2.imwrite('{}/{}_input_noisy.jpg'.format(image_path_input_noisy, i), img_noisy_np_print[:, :, i])

toc = time.perf_counter()

print(f"Training the network in {toc - tic:0.4f} seconds")
