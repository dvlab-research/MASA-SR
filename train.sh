# train masa-rec only with the reconstruction loss
python train.py --use_tb_logger --data_augmentation --max_iter 160 --loss_l1 --name train_masa_rec

# # train masa with all losses, this model is based on the pretrained masa-rec
# python train.py --use_tb_logger --max_iter 50 --loss_l1 --loss_adv --loss_perceptual --name train_masa_gan --resume ./weights/train_masa_rec/snapshot/net_best.pth --resume_optim ./weights/train_masa_rec/snapshot/optimizer_G_best.pth --resume_scheduler ./weights/train_masa_rec/snapshot/scheduler_best.pth
