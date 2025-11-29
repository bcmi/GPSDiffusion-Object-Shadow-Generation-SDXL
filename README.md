# GPSDiffusion-Object-Shadow-Generation-SDXL

This repository presents GPSDiffusion-SDXL, an upgraded version of our [GPSDiffusion](https://github.com/bcmi/GPSDiffusion-Object-Shadow-Generation):

> **Shadow Generation Using Diffusion Model with Geometry Prior** [[pdf]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhao_Shadow_Generation_Using_Diffusion_Model_with_Geometry_Prior_CVPR_2025_paper.pdf) [[supp]](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Zhao_Shadow_Generation_Using_CVPR_2025_supplemental.pdf) <br>  
>
> Haonan Zhao, Qingyang Liu, Xinhao Tao, Li Niu, Guangtao Zhai<br>
> Accepted by **CVPR 2025**.

We replace the original Stable Diffusion 1.5 (SD 1.5) with the more advanced Stable Diffusion XL (SDXL) for enhanced generation performance. The following visual comparison demonstrates the quality improvements of SDXL over its predecessor SD 1.5. From left to right, we show the composite image, foreground mask, the result based on SD 1.5, the result based on SD XL, and ground-truth.

<p align='center'>  
  <img src='cmp_with_sd1_5.png'  width=80% />
</p>

We also present a visual comparison between SDXL and SD 1.5 using different random seeds. From left to right, we show the composite image, foreground mask, four SD1.5 outputs with varying seeds, four SDXL outputs with varying seeds, and the ground-truth. Notably, the SDXL version demonstrates significantly improved output stability.

<p align='center'>  
  <img src='cmp_with_sd1_5_ranseed.png'  width=90% />
</p>


### Online Demo

Try this [online demo](http://libcom.ustcnewly.com/) for image composition and have fun!

### Installation
- Clone this repo:
    git clone https://github.com/bcmi/GPSDiffusion-Object-Shadow-Generation-SDXL.git
- Download the DESOBAv2 dataset from [[Baidu Cloud]](https://pan.baidu.com/s/1_nXb3ElxImmsq2BPcBGdPQ?pwd=bcmi) (access code: bcmi) or [[Dropbox]](https://www.dropbox.com/scl/fo/f71dg98aszqxtn2qs3l1c/ALS7dpAe3dBPbYbRaq10mnY?rlkey=6cm1vcma91yn06ziy3v4cxzxg&st=69kd9ihx&dl=0). Unzip `desobav2-256x256.rar` to `./data/`, and rename it to `desobav2`.
- Download the checkpoints from [[Baidu Cloud]](https://pan.baidu.com/s/13NYGw3SS4B4n6mPtU1Me1Q?pwd=bcmi) (access code: bcmi). Unzip `pretrained_models.zip`.

### Environment
    conda create -n GPSDiffusion python=3.8
    conda activate GPSDiffusion
    pip install -r requirements.txt

    git clone https://github.com/huggingface/diffusers
    cd diffusers
    pip install -e .

And initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

    accelerate config

Or for a default accelerate configuration without answering questions about your environment:

    accelerate config default

### Training
    accelerate launch train_GPSDiffusion_sdxl.py

### Inference
    accelerate launch test_GPSDiffusion_sdxl.py

### Post-processing
    python post_processing.py
    
## Other Resources

+ [Awesome-Object-Shadow-Generation](https://github.com/bcmi/Awesome-Object-Shadow-Generation)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion)

## Acknowledgments
Parts of this code were derived from:<br>
https://github.com/huggingface/diffusers
