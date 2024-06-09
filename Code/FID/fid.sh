# based on https://github.com/mseitzer/pytorch-fid
#images must be resized to 256,256

source activate Torch
export PATH="/users/PCS0218/nonlinearity114/.local/bin:$PATH"

DDPM_Normal=/users/PCS0218/nonlinearity114/CS7200_SP2024_Project_G01/Notebooks/VGG16-classification/DDPM_Generated/NORMAL
DDPM_Pneumonia=/users/PCS0218/nonlinearity114/CS7200_SP2024_Project_G01/Notebooks/VGG16-classification/DDPM_Generated/PNEUMONIA

PG150_Normal=/users/PCS0218/nonlinearity114/PGAN/Images512ch150/NORMAL
PG150_Pneumonia=/users/PCS0218/nonlinearity114/PGAN/Images512ch150/PNEUMONIA

PG160_Normal=/users/PCS0218/nonlinearity114/PGAN/Images512ch160/NORMAL
PG160_Pneumonia=/users/PCS0218/nonlinearity114/PGAN/Images512ch160/PNEUMONIA

Original_Normal=/users/PCS0218/nonlinearity114/CS7200_SP2024_Project_G01/resized/original/Normal
Original_Pneumonia=/users/PCS0218/nonlinearity114/CS7200_SP2024_Project_G01/resized/original/PNEUMONIA

pytorch-fid $Original_Normal $DDPM_Normal --device cuda:0
pytorch-fid $Original_Pneumonia $DDPM_Pneumonia --device cuda:0

pytorch-fid $Original_Normal $PG150_Normal --device cuda:0
pytorch-fid $Original_Pneumonia $PG150_Pneumonia --device cuda:0


pytorch-fid $Original_Normal $PG160_Normal --device cuda:0
pytorch-fid $Original_Pneumonia $PG160_Pneumonia --device cuda:0
