import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class AFM_E(nn.Module):
    def __init__(self, fq_bound=1.0):
        super().__init__()
        torch.use_deterministic_algorithms(True)

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.dconv_down1 = double_conv(12, 32)  # 256x128
        self.dconv_down2 = double_conv(32, 64)          # 128x64
        self.dconv_down3 = double_conv(64, 128)         # 64x32
        self.dconv_down4 = double_conv(128, 256)         # 32x16

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up3 = double_conv(128 + 256, 128)     # 64x32
        self.dconv_up2 = double_conv(64 + 128, 64)     # 128x64
        self.dconv_up1 = double_conv(32 + 64, 32)       # 256x128

        self.conv_last = nn.Conv2d(32, 1, 1)  # 256x128
        self.sigmoid = nn.Sigmoid()
        self.fq_bound = fq_bound

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

   
    def forward(self, clean, noisy):
        B, C, H, W = noisy.size()
  
        noisy_fq = torch.fft.fftn(noisy, dim=(-1,-2))
        noisy_fq = torch.fft.fftshift(noisy_fq)
        clean_fq = torch.fft.fftn(clean, dim=(-1,-2))
        clean_fq = torch.fft.fftshift(clean_fq)

        # filter_input = torch.cat([torch.real(half_noise_fq), torch.real(half_clean_fq), torch.imag(half_noise_fq), torch.imag(half_clean_fq)], dim=1) 
        filter_input = torch.cat([noisy, torch.log10(torch.abs(noisy_fq) + 1), clean, torch.log10(torch.abs(clean_fq) + 1)], dim=1)
       
        conv1 = self.dconv_down1(filter_input)     # [8, 32, 256, 256] #c3
        x = self.maxpool(conv1)         # [8, 32, 128, 128]

        conv2 = self.dconv_down2(x)     # [8, 64, 128, 128] #c2
        x = self.maxpool(conv2)         # [8, 64, 64, 64]

        conv3 = self.dconv_down3(x)     # [8, 128, 64, 64] #c1
        x = self.maxpool(conv3)         # [8, 128, 32, 32]

        x = self.dconv_down4(x)         # [8, 256, 32, 32]
        x = self.upsample(x)            # [8, 256, 64, 64] #c1
        x = torch.cat([x, conv3], dim=1)  # [8, 128+256, 64, 64]

        x = self.dconv_up3(x)           # [8, 128, 64, 64]
        x = self.upsample(x)            # [8, 128, 128, 128] #c2
        x = torch.cat([x, conv2], dim=1)  # [8, 64+128, 128, 128]

        x = self.dconv_up2(x)           # [8, 64, 128, 128]
        x = self.upsample(x)            # [8, 64, 256, 256] #c3
        x = torch.cat([x, conv1], dim=1)  # [8, 32+64, 256, 256]

        x = self.dconv_up1(x)           # [8, 32, 256, 256]

        x = self.conv_last(x)         # [8, 3, 256, 256]
        out = self.sigmoid(x)          # [8, 3, 256, 256]

        half_out = out[:, :, :, :128]
        flipped_out_horizontal = torch.flip(half_out, dims=[3])
        flipped_out_vertical = torch.flip(flipped_out_horizontal, dims=[2])
        fq_mask = torch.cat([half_out, flipped_out_vertical], dim=3)
        fq_mask = fq_mask*self.fq_bound
        fq_mask_rev = torch.ones_like(fq_mask)-fq_mask

        replaced_fq_hard = noisy_fq*fq_mask+clean_fq*fq_mask_rev
        replaced_fq_hard = torch.fft.ifftshift(replaced_fq_hard)
        replaced_fq_hard = torch.fft.ifftn(replaced_fq_hard, dim=(-1,-2))
        replaced_fq_hard = replaced_fq_hard.real

        replaced_fq_easy = noisy_fq*fq_mask_rev+clean_fq*fq_mask
        replaced_fq_easy = torch.fft.ifftshift(replaced_fq_easy)
        replaced_fq_easy = torch.fft.ifftn(replaced_fq_easy, dim=(-1,-2))
        replaced_fq_easy = replaced_fq_easy.real

        value_set = fq_mask
        return replaced_fq_hard, replaced_fq_easy, value_set
    

