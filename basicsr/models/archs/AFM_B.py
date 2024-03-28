import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class AFM_B(nn.Module):
    def __init__(self, fq_bound=1.0):
        super().__init__()

        self.radius_factor_set = torch.arange(0.01, 1.01, 0.01).cuda()

        self.fq_bound = fq_bound
        self.sigmoid = nn.Sigmoid()
        

        self.conv1 = nn.Conv2d(in_channels=12, out_channels=16, kernel_size=3, padding=1, stride=1, groups=1, bias=True).cuda()
        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1, groups=1,bias=True).cuda()
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1, groups=1, bias=True).cuda()
        
        self.dropout = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=-1)
        self.fclayer_v1 = nn.Linear(64, 256).cuda()
        self.fclayer_last = nn.Linear(256, len(self.radius_factor_set)*len(self.radius_factor_set)).cuda()
        self.leaky_relu = nn.LeakyReLU()
        self.temperature = 0.1

        # tmp
        self.value_set = 0.

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

   
    def forward(self, clean, noisy):
        B, C, H, W = noisy.size()
        inp = noisy

        a, b = torch.meshgrid(torch.arange(H), torch.arange(W))
        dist = torch.sqrt((a - H/2)**2 + (b - W/2)**2)
        dist = dist.to(noisy.device)
        max_radius = math.sqrt(H*H+W*W)/2

       
        noisy_fq = torch.fft.fftn(noisy, dim=(-1,-2))
        noisy_fq = torch.fft.fftshift(noisy_fq)
        clean_fq = torch.fft.fftn(clean, dim=(-1,-2))
        clean_fq = torch.fft.fftshift(clean_fq)


        ####################################################
        filter_input = torch.cat([noisy, torch.log10(torch.abs(noisy_fq) + 1), clean, torch.log10(torch.abs(clean_fq) + 1)], dim=1)
        # filter_input = torch.cat([noisy, clean], dim=1)
        y = self.conv1(filter_input)
        y = self.relu(y)
        y = self.down1(y)
        y = self.conv2(y)
        y = self.relu(y)
        y = self.down2(y)
        y = self.conv3(y)
        y = self.relu(y)

        y = self.avgpool(y)
        y = y.squeeze(-1)
        y = y.squeeze(-1)

        # using softmax
        value_prob =  self.fclayer_last(self.fclayer_v1(y))
        value_prob = value_prob.view(B, 100, 100)
        value_prob = self.soft(value_prob * self.temperature ) * (self.radius_factor_set.unsqueeze(0).unsqueeze(1))
        value_prob = value_prob.sum(dim=-1)
        value_prob = value_prob.squeeze(-1)
        value_set = (value_prob*self.fq_bound).cuda()
        

        radius_set = max_radius*self.radius_factor_set


        mask = []
        zero_mask = torch.zeros_like(dist).cuda()
        one_mask = torch.ones_like(dist).cuda()
        for i in range(len(radius_set)):
            if i == 0:
                mask.append(torch.where((dist < radius_set[i]), one_mask, zero_mask))
            else :
                mask.append(torch.where((dist < radius_set[i]) & (dist >= radius_set[i-1]), one_mask, zero_mask))
           

        fq_mask_set = torch.stack(mask, dim=0)
        fq_mask = value_set.unsqueeze(-1).unsqueeze(-1) * fq_mask_set.unsqueeze(0)
        fq_mask = torch.sum(fq_mask, dim=1)

    

        bn1_mask = fq_mask
        bn2_mask = torch.ones_like(bn1_mask)-bn1_mask

        noisy_fq_hard = (noisy_fq*bn1_mask.unsqueeze(1))
        clean_fq_hard = (clean_fq*bn2_mask.unsqueeze(1))
        replaced_fq_hard = noisy_fq_hard+clean_fq_hard
        replaced_fq_hard = torch.fft.ifftshift(replaced_fq_hard)
        replaced_fq_hard = torch.fft.ifftn(replaced_fq_hard, dim=(-1,-2))
        replaced_fq_hard = replaced_fq_hard.real

        noisy_fq_easy = (noisy_fq*bn2_mask.unsqueeze(1))
        clean_fq_easy = (clean_fq*bn1_mask.unsqueeze(1))
        replaced_fq_easy = noisy_fq_easy+clean_fq_easy
        replaced_fq_easy = torch.fft.ifftshift(replaced_fq_easy)
        replaced_fq_easy = torch.fft.ifftn(replaced_fq_easy, dim=(-1,-2))
        replaced_fq_easy = replaced_fq_easy.real


        return replaced_fq_hard, replaced_fq_easy, fq_mask