<!-- The official pytorch implementation of the paper **[Simple Baselines for Image Restoration (ECCV2022)](https://arxiv.org/abs/2204.04676)** -->

# Robust Image Denoising through Adversarial Frequency Mixup (2024 CVPR) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Ryou_Robust_Image_Denoising_through_Adversarial_Frequency_Mixup_CVPR_2024_paper.html)


## Installation
This implementation based on [BasicSR](https://github.com/xinntao/BasicSR) which is a open source toolbox for image/video restoration tasks and [NAFNet](https://github.com/megvii-research/NAFNet) 

```python
python 3.8.8
pytorch 1.9.0
cuda 11.3
```

```
git clone https://github.com/dhryougit/AFM.git
cd AFM
pip install -r requirements.txt
python setup.py develop --no_cuda_ext
```

We used NVIDIA RTX A6000 D6 48GB for trianing our models.<br><br>


## QuickStart
For training 
```
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/DnCNN.yml --name=DnCNN-afm-b --afm_type=AFM_B --seed=10 --afm_rate=0.8 --afm_easy_rate=0.3 --launcher pytorch
```
<br>

For test
```
python3 -m torch.distributed.launch --nproc_per_node=1 basicsr/test.py -opt options/test/DnCNN.yml -name=AFM_test --launcher pytorch
```
<br>

## Dataset

Training dataset : [SIDD](https://abdokamel.github.io/sidd/#sidd-medium)

Evaluation datasets : [Poly](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset), [CC](https://github.com/csjunxu/MCWNNM-ICCV2017), HighISO, iPhone, Huawei.

Additioanl real-world noise datasets can be downloaded from "https://github.com/ZhaomingKong/Denoising-Comparison"<br><br>



## Results and Pre-trained model


| Dataset | Poly |CC |HighISO |iPhone |Huawei | OOD Avg.|
|:----|:----|:----|:----|:----|:----|-----|
|PSRN| 37.75  | 36.84 | 39.17   | 40.65   | 38.39   | 38.56   |
|SSIM| 0.9804 | 0.9830 | 0.9801  | 0.9777  | 0.9683  | 0.9779  |

Pre-trained model of our Dncnn trained on AFM-B can be downloaded from (https://drive.google.com/file/d/1uPJP2zNc4ViFc1QU7TXGAwrFIEScBJvL/view?usp=sharing)
