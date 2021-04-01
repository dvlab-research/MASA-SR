# test the masa_rec model (trained with only reconstruction loss)
python test.py --resume './pretrained_weights/masa_rec.pth' --testset TestSet_multi --name masa_rec_TestSet_multi  # test on CUFED5 testing set, using multiple refs.
# python test.py --resume './pretrained_weights/masa_rec.pth' --testset TestSet --name masa_rec_TestSet --ref_level 1  # test on CUFED5 testing set, using single ref.
# python test.py --resume './pretrained_weights/masa_rec.pth' --testset Sun80 --name masa_rec_Sun80  # test on Sun80 dataset.
# python test.py --resume './pretrained_weights/masa_rec.pth' --testset Urban100 --name masa_rec_Urban100  # test on Urban100 dataset.

# # test the masa model (trained with all losses)
# python test.py --resume './pretrained_weights/masa.pth' --testset TestSet_multi --name masa_TestSet_multi  # test on CUFED5 testing set, using multiple refs.
# python test.py --resume './pretrained_weights/masa.pth' --testset TestSet --name masa_TestSet --ref_level 1  # test on CUFED5 testing set, using single ref.
# python test.py --resume './pretrained_weights/masa.pth' --testset Sun80 --name masa_Sun80  # test on Sun80 dataset.
# python test.py --resume './pretrained_weights/masa.pth' --testset Urban100 --name masa_Urban100  # test on Urban100 dataset.