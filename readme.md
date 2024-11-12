<!-- The official pytorch implementation of the paper **[Simple Baselines for Image Restoration (ECCV2022)](https://arxiv.org/abs/2204.04676)** -->

# Robust Image Denoising through Adversarial Frequency Mixup (2024 CVPR) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Ryou_Robust_Image_Denoising_through_Adversarial_Frequency_Mixup_CVPR_2024_paper.html)

### Installation
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

We used NVIDIA RTX A6000 D6 48GB for trianing our models.

### QuickStart
For training 
```
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/DnCNN.yml --name=DnCNN-afm-b --afm_type=AFM_B --seed=10 --afm_rate=0.8 --afm_easy_rate=0.3 --launcher pytorch
```


### Dataset

Training dataset : [SIDD](https://abdokamel.github.io/sidd/#sidd-medium)

Evaluation datasets : [Poly](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset), [CC](https://github.com/csjunxu/MCWNNM-ICCV2017), HighISO, iPhone, Huawei.

Additioanl real-world noise datasets can be downloaded from "https://github.com/ZhaomingKong/Denoising-Comparison"


### Results and Pre-trained model


| Dataset | Poly |CC |HighISO |iPhone |Huawei | OOD Avg.|
|:----|:----|:----|:----|:----|:----|-----|
|PSRN| 37.75  | 36.84 | 39.17   | 40.65   | 38.39   | 38.56   |
|SSIM| 0.9804 | 0.9830 | 0.9801  | 0.9777  | 0.9683  | 0.9779  |

Pre-trained model of our Dncnn trained on AFM-B can be downloaded from (https://drive.google.com/file/d/1uPJP2zNc4ViFc1QU7TXGAwrFIEScBJvL/view?usp=sharing)
<!-- * Image Denoise Colab Demo: [<a href="https://colab.research.google.com/drive/1dkO5AyktmBoWwxBwoKFUurIDn0m4qDXT?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1dkO5AyktmBoWwxBwoKFUurIDn0m4qDXT?usp=sharing)
* Image Deblur Colab Demo: [<a href="https://colab.research.google.com/drive/1yR2ClVuMefisH12d_srXMhHnHwwA1YmU?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1yR2ClVuMefisH12d_srXMhHnHwwA1YmU?usp=sharing)
* Stereo Image Super-Resolution Colab Demo: [<a href="https://colab.research.google.com/drive/1PkLog2imf7jCOPKq1G32SOISz0eLLJaO?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1PkLog2imf7jCOPKq1G32SOISz0eLLJaO?usp=sharing)
* Single Image Inference Demo:
    * Image Denoise:
    ```
    python basicsr/demo.py -opt options/test/SIDD/NAFNet-width64.yml --input_path ./demo/noisy.png --output_path ./demo/denoise_img.png
  ```
    * Image Deblur:
    ```
    python basicsr/demo.py -opt options/test/REDS/NAFNet-width64.yml --input_path ./demo/blurry.jpg --output_path ./demo/deblur_img.png
    ```
    * ```--input_path```: the path of the degraded image
    * ```--output_path```: the path to save the predicted image
    * [pretrained models](https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models) should be downloaded. 
    * Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo for single image restoration[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/chuxiaojie/NAFNet)
* Stereo Image Inference Demo:
    * Stereo Image Super-resolution:
    ```
    python basicsr/demo_ssr.py -opt options/test/NAFSSR/NAFSSR-L_4x.yml \
    --input_l_path ./demo/lr_img_l.png --input_r_path ./demo/lr_img_r.png \
    --output_l_path ./demo/sr_img_l.png --output_r_path ./demo/sr_img_r.png
    ```
    * ```--input_l_path```: the path of the degraded left image
    * ```--input_r_path```: the path of the degraded right image
    * ```--output_l_path```: the path to save the predicted left image
    * ```--output_r_path```: the path to save the predicted right image
    * [pretrained models](https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models) should be downloaded. 
    * Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the Web Demo for stereo image super-resolution[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/chuxiaojie/NAFSSR)
* Try the web demo with all three tasks here: [![Replicate](https://replicate.com/megvii-research/nafnet/badge)](https://replicate.com/megvii-research/nafnet) -->

<!-- ### Results and Pre-trained Models -->
<!-- 
| name | Dataset|PSNR|SSIM| pretrained models | configs |
|:----|:----|:----|:----|:----|-----|
|NAFNet-GoPro-width32|GoPro|32.8705|0.9606|[gdrive](https://drive.google.com/file/d/1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1AbgG0yoROHmrRQN7dgzDvQ?pwd=so6v)|[train](./options/train/GoPro/NAFNet-width32.yml) \| [test](./options/test/GoPro/NAFNet-width32.yml)|
|NAFNet-GoPro-width64|GoPro|33.7103|0.9668|[gdrive](https://drive.google.com/file/d/1S0PVRbyTakYY9a82kujgZLbMihfNBLfC/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1g-E1x6En-PbYXm94JfI1vg?pwd=wnwh)|[train](./options/train/GoPro/NAFNet-width64.yml) \| [test](./options/test/GoPro/NAFNet-width64.yml)|
|NAFNet-SIDD-width32|SIDD|39.9672|0.9599|[gdrive](https://drive.google.com/file/d/1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1Xses38SWl-7wuyuhaGNhaw?pwd=um97)|[train](./options/train/SIDD/NAFNet-width32.yml) \| [test](./options/test/SIDD/NAFNet-width32.yml)|
|NAFNet-SIDD-width64|SIDD|40.3045|0.9614|[gdrive](https://drive.google.com/file/d/14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/198kYyVSrY_xZF0jGv9U0sQ?pwd=dton)|[train](./options/train/SIDD/NAFNet-width64.yml) \| [test](./options/test/SIDD/NAFNet-width64.yml)|
|NAFNet-REDS-width64|REDS|29.0903|0.8671|[gdrive](https://drive.google.com/file/d/14D4V4raNYIOhETfcuuLI3bGLB-OYIv6X/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1vg89ccbpIxg3mK9IONBfGg?pwd=9fas)|[train](./options/train/REDS/NAFNet-width64.yml) \| [test](./options/test/REDS/NAFNet-width64.yml)|
|NAFSSR-L_4x|Flickr1024|24.17|0.7589|[gdrive](https://drive.google.com/file/d/1TIdQhPtBrZb2wrBdAp9l8NHINLeExOwb/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1P8ioEuI1gwydA2Avr3nUvw?pwd=qs7a)|[train](./options/test/NAFSSR/NAFSSR-L_4x.yml) \| [test](./options/test/NAFSSR/NAFSSR-L_4x.yml)|
|NAFSSR-L_2x|Flickr1024|29.68|0.9221|[gdrive](https://drive.google.com/file/d/1SZ6bQVYTVS_AXedBEr-_mBCC-qGYHLmf/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1GS6YQSSECH8hAKhvzw6GyQ?pwd=2v3v)|[train](./options/test/NAFSSR/NAFSSR-L_2x.yml) \| [test](./options/test/NAFSSR/NAFSSR-L_2x.yml)|
|Baseline-GoPro-width32|GoPro|32.4799|0.9575|[gdrive](https://drive.google.com/file/d/14z7CxRzVkYEhFgsZg79GlPTEr3VFIGyl/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1WnFKYTAQyAQ9XuD5nlHw_Q?pwd=oieh)|[train](./options/train/GoPro/Baseline-width32.yml) \| [test](./options/test/GoPro/Baseline-width32.yml)|
|Baseline-GoPro-width64|GoPro|33.3960|0.9649|[gdrive](https://drive.google.com/file/d/1yy0oPNJjJxfaEmO0pfPW_TpeoCotYkuO/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1Fqi2T4nyF_wo4wh1QpgIGg?pwd=we36)|[train](./options/train/GoPro/Baseline-width64.yml) \| [test](./options/test/GoPro/Baseline-width64.yml)|
|Baseline-SIDD-width32|SIDD|39.8857|0.9596|[gdrive](https://drive.google.com/file/d/1NhqVcqkDcYvYgF_P4BOOfo9tuTcKDuhW/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1wkskmCRKhXq6dGa6Ns8D0A?pwd=0rin)|[train](./options/train/SIDD/Baseline-width32.yml) \| [test](./options/test/SIDD/Baseline-width32.yml)|
|Baseline-SIDD-width64|SIDD|40.2970|0.9617|[gdrive](https://drive.google.com/file/d/1wQ1HHHPhSp70_ledMBZhDhIGjZQs16wO/view?usp=sharing)  \|  [百度网盘](https://pan.baidu.com/s/1ivruGfSRGfWq5AEB8qc7YQ?pwd=t9w8)|[train](./options/train/SIDD/Baseline-width64.yml) \| [test](./options/test/SIDD/Baseline-width64.yml)| -->


<!-- ### Image Restoration Tasks  -->
<!-- 
| Task                                 | Dataset | Train/Test Instructions            | Visualization Results                                        |
| :----------------------------------- | :------ | :---------------------- | :----------------------------------------------------------- |
| Image Deblurring                     | GoPro   | [link](./docs/GoPro.md) | [gdrive](https://drive.google.com/file/d/1S8u4TqQP6eHI81F9yoVR0be-DLh4cNgb/view?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/1yNYQhznChafsbcfHO44aHQ?pwd=96ii)|
| Image Denoising                      | SIDD    | [link](./docs/SIDD.md)  | [gdrive](https://drive.google.com/file/d/1rbBYD64bfvbHOrN3HByNg0vz6gHQq7Np/view?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/1wIubY6SeXRfZHpp6bAojqQ?pwd=hu4t)|
| Image Deblurring with JPEG artifacts | REDS    | [link](./docs/REDS.md)  | [gdrive](https://drive.google.com/file/d/1FwHWYPXdPtUkPqckpz-WBitpVyPuXFRi/view?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/17T30w5xAtBQQ2P3wawLiVA?pwd=put5) |
| Stereo Image Super-Resolution | Flickr1024+Middlebury    | [link](./docs/StereoSR.md)  | [gdrive](https://drive.google.com/drive/folders/1lTKe2TU7F-KcU-oaF8jqgoUwIMb6RW0w?usp=sharing)   \|   [百度网盘](https://pan.baidu.com/s/1kov6ivrSFy1FuToCATbyrA?pwd=q263 ) | -->


<!-- ### Citations
If NAFNet helps your research or work, please consider citing NAFNet.

```
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
```
If NAFSSR helps your research or work, please consider citing NAFSSR.
```
@InProceedings{chu2022nafssr,
    author    = {Chu, Xiaojie and Chen, Liangyu and Yu, Wenqing},
    title     = {NAFSSR: Stereo Image Super-Resolution Using NAFNet},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1239-1248}
}
``` -->

<!-- ### Contact

If you have any questions, please contact chenliangyu@megvii.com or chuxiaojie@megvii.com

---

<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.glitch.me/badge?page_id=megvii-research/NAFNet)

</details>
 -->
