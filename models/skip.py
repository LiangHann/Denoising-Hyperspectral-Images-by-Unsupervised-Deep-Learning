import torch
import torch.nn as nn
from .common import *

class skip(nn.Module):
    def __init__(self, num_input_channels=2, num_output_channels=3, 
                 num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
                 filter_size_down=3, filter_size_up=3, filter_skip_size=1,
                 need_sigmoid=True, need_bias=True, 
                 pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', need1x1_up=True):
        super(skip, self).__init__()
        """Assembles encoder-decoder with skip connections.

        Arguments:
            act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
            pad (string): zero|reflection (default: 'zero')
            upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
            downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

        """
        assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

        self.n_scales = len(num_channels_down) 

        if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
            self.upsample_mode = [upsample_mode] * self.n_scales

        if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
            self.downsample_mode = [downsample_mode] * self.n_scales
    
        if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
            self.filter_size_down = [filter_size_down] * self.n_scales

        if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
            self.filter_size_up = [filter_size_up] * self.n_scales

        self.last_scale = self.n_scales - 1 

        self.cur_depth = None

        self.Sigmoid = nn.Sigmoid()

        self.model = nn.Sequential()
        self.model_tmp = self.model

        self.input_depth = num_input_channels
        for i in range(len(num_channels_down)):

            self.deeper = nn.Sequential()
            self.skip = nn.Sequential()

            if num_channels_skip[i] != 0:
                self.model_tmp.add(Concat(1, self.skip, self.deeper))
            else:
                self.model_tmp.add(self.deeper)
        
            self.model_tmp.add(bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < self.last_scale else num_channels_down[i])))

            if num_channels_skip[i] != 0:
                self.skip.add(conv(self.input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
                self.skip.add(bn(num_channels_skip[i]))
                self.skip.add(act(act_fun))
            

            self.deeper.add(conv(self.input_depth, num_channels_down[i], self.filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=self.downsample_mode[i]))
            self.deeper.add(bn(num_channels_down[i]))
            self.deeper.add(act(act_fun))

            self.deeper.add(conv(num_channels_down[i], num_channels_down[i], self.filter_size_down[i], bias=need_bias, pad=pad))
            self.deeper.add(bn(num_channels_down[i]))
            self.deeper.add(act(act_fun))

            self.deeper_main = nn.Sequential()

            if i == len(num_channels_down) - 1:
                # The deepest
                k = num_channels_down[i]
            else:
                self.deeper.add(self.deeper_main)
                k = num_channels_up[i + 1]

            self.deeper.add(nn.Upsample(scale_factor=2, mode=self.upsample_mode[i]))

            self.model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], self.filter_size_up[i], 1, bias=need_bias, pad=pad))
            self.model_tmp.add(bn(num_channels_up[i]))
            self.model_tmp.add(act(act_fun))


            if need1x1_up:
                self.model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
                self.model_tmp.add(bn(num_channels_up[i]))
                self.model_tmp.add(act(act_fun))

            self.input_depth = num_channels_down[i]
            self.model_tmp = self.deeper_main

        self.model_final = nn.Sequential()
        self.model_final.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
        if need_sigmoid:
            self.model_final.add(nn.Sigmoid())

    def forward(self, x):
        for i in range(len(self.model)):
            x = self.model[i](x)
        output = self.model_final(x)
        return output, self.Sigmoid(x)

