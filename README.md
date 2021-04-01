# MASA-SR
Official PyTorch implementation of our CVPR2021 paper [MASA-SR: Matching Acceleration and Spatial Adaptation for Reference-Based Image Super-Resolution](https://jiaya.me/papers/masasr_cvpr21.pdf)



## Dependencies
* python 3
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

    * Pre-trained models can be downloaded from [onedrive](https://bit.ly/3dqo1Mk)
        * *masa_rec.pth*: trained with only reconstruction loss
        * *masa.pth*: trained with all losses
1. Run test.sh. See more details in test.sh (if you are using cpu, please add `--gpu_ids -1` in the command)
    ```
    sh test.sh
    ```
1. The testing results are in the `test_results/` folder

## Train
1. First train masa-rec only with the reconstruction loss.
    ```
    python train.py --use_tb_logger --data_augmentation --max_iter 160 --loss_l1 --name train_masa_rec
    ```
1. After getting masa-rec, train masa with all losses, which is based on the pretrained masa-rec.
    ```
    python train.py --use_tb_logger --max_iter 50 --loss_l1 --loss_adv --loss_perceptual --name train_masa_gan --resume ./weights/train_masa_rec/snapshot/net_best.pth --resume_optim ./weights/train_masa_rec/snapshot/optimizer_G_best.pth --resume_scheduler ./weights/train_masa_rec/snapshot/scheduler_best.pth
    ```
1. The training results are in the `weights/` folder
