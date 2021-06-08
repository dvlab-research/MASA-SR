# MASA-SR
Official PyTorch implementation of our CVPR2021 paper [MASA-SR: Matching Acceleration and Spatial Adaptation for Reference-Based Image Super-Resolution](https://arxiv.org/abs/2106.02299)



## Dependencies
* python >= 3.5
* pytorch >= 1.1.0
* torchvision >= 0.4.0

## Prepare Dataset 
1. Download [CUFED train set](https://drive.google.com/drive/folders/1hGHy36XcmSZ1LtARWmGL5OK1IUdWJi3I) and [CUFED test set](https://drive.google.com/file/d/1Fa1mopExA9YGG1RxrCZZn7QFTYXLx6ph/view)
1. Place the datasets in this structure:
    ```
    CUFED
    ├── train
    │   ├── input
    │   └── ref 
    └── test
        └── CUFED5  
    ```
## Get Started
1. Clone this repo
    ```
    git clone https://github.com/Jia-Research-Lab/MASA-SR.git
    cd MASA-SR
    ```
1. Download the dataset. Modify the argument `--data_root` in `test.py` and `train.py` according to your data path.
### Evaluation
1. Download the pre-trained models and place them into the `pretrained_weights/` folder

    * Pre-trained models can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1O9Y4UK1PFoFKOsYTQqcQJCA_VvBWp29N?usp=sharing)
        * *masa_rec.pth*: trained with only reconstruction loss
        * *masa.pth*: trained with all losses
1. Run test.sh. See more details in test.sh (if you are using cpu, please add `--gpu_ids -1` in the command)
    ```
    sh test.sh
    ```
1. The testing results are in the `test_results/` folder

### Training
1. First train masa-rec only with the reconstruction loss.
    ```
    python train.py --use_tb_logger --data_augmentation --max_iter 250 --loss_l1 --name train_masa_rec
    ```
1. After getting masa-rec, train masa with all losses, which is based on the pretrained masa-rec.
    ```
    python train.py --use_tb_logger --max_iter 50 --loss_l1 --loss_adv --loss_perceptual --name train_masa_gan --resume ./weights/train_masa_rec/snapshot/net_best.pth --resume_optim ./weights/train_masa_rec/snapshot/optimizer_G_best.pth --resume_scheduler ./weights/train_masa_rec/snapshot/scheduler_best.pth
    ```
1. The training results are in the `weights/` folder

## Update
[2021/06/08] Fix a bug in evaluation. Retrain the models and update the given checkpoints, whose PSNR have a slight difference with those reported in the paper (±0.03dB).

## Acknowledgement
We borrow some codes from [TTSR](https://github.com/researchmm/TTSR) and [BasicSR](https://github.com/xinntao/BasicSR). We thank the authors for their great work.

## Citation

Please consider citing our paper in your publications if it is useful for your research.
```
@inproceedings{lu2021masasr,
    title={MASA-SR: Matching Acceleration and Spatial Adaptation for Reference-Based Image Super-Resolution},
    author={Liying Lu, Wenbo Li, Xin Tao, Jiangbo Lu, and Jiaya Jia},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2021},
}
```