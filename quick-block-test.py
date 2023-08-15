'''quick version of block patch prediction, set batchsize = 128, seamlessly stitch image patches.'''
import time
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import pandas as pd
from PIL import Image
import cv2
from torch.optim.lr_scheduler import StepLR
import albumentations as A
import utils.recompose as rp
from utils.ihc_data import PanNukeDataModule
from utils.SRSANet import SRSANet, loss_hovernet, post_process_batch_hovernet
from pathml.ml.utils import wrap_transform_multichannel, dice_score
from pathml.utils import plot_segmentation,segmentation_lines
from torchvision.transforms import transforms
# Function to load Images and Masks
start = time.time()
def get_images(parent_dir):
    tissue_dir = parent_dir
    im_names = os.listdir(tissue_dir)
    X = np.zeros((len(im_names),block_height,block_weight, 3), dtype=np.float32)
    # y = np.zeros((len(im_names), im_height, im_width, 1), dtype=np.float32)
    id = 1
    for im_name in im_names:
    # Load images
        img = im_name
        img_path = os.path.join(tissue_dir,img)
        x_img = cv2.imread(img_path)
        # x_img = cv2.resize(x_img, (block_height,block_weight))
        X[id-1] = x_img
        id += 1
        # y[n] = mask/255.0
    return X,im_names


# In[4]:
block_height=10240
block_weight=10240

# Load the training images and mask
X_test,im_names = get_images("./block_result/image_block3/")
patches_imgs_test, extended_height, extended_width = rp.get_data_testing_overlap(
        X_test,  # original
        n_test_images=3,
        patch_height=256,
        patch_width=256,
        stride_height=128,
        stride_width=128,
        channel=3)


#training with multi GPUs
print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda:0")
print(f"Device:\t\t{device}")

n_classes_pannuke = 3

# load the model
srsanet = SRSANet(n_classes=n_classes_pannuke)

# wrap model to use multi-GPU
srsanet = torch.nn.DataParallel(srsanet)

# set up optimizer
opt = torch.optim.Adam(srsanet.parameters(), lr = 1e-4)
# learning rate scheduler to reduce LR by factor of 10 each 25 epochs
scheduler = StepLR(opt, step_size=25, gamma=0.1)

# send model to GPU
srsanet.to(device)


# load the best model
checkpoint = torch.load("srsanet_best_perf.pt")
srsanet.load_state_dict(checkpoint)
srsanet.eval()
image = None
mask_pred = None
batchsize = 64
images = np.zeros((patches_imgs_test.shape[0]//batchsize+1,batchsize,3,256,256))
i = 0
while i < patches_imgs_test.shape[0]:
    d = i//batchsize
    for j in range(batchsize):
        if j == 0:
            image = patches_imgs_test[j+d*batchsize,:,:,:]
            image = image[np.newaxis, :]
        else:
            if j+d*batchsize < patches_imgs_test.shape[0]:
                data = patches_imgs_test[j+d*batchsize,:,:,:]
                data = data[np.newaxis, :]
            else:
                data = np.zeros((1,3,256,256))
            image = np.concatenate([image, data])
    images[d,:,:,:,:] = image
    i+=batchsize
with torch.no_grad():
    for i, data in tqdm(enumerate(images)):
        data = torch.tensor(data)
        data = data.float().to(device)
        # pass thru network to get predictions
        outputs = srsanet(data)
        preds_detection, preds_classification = post_process_batch_hovernet(outputs, n_classes=3)

        if i == 0:
            mask_pred = preds_classification
        else:
            mask_pred = np.concatenate([mask_pred, preds_classification], axis=0)

    # collapse multi-class preds into binary preds
preds_detection = np.sum(mask_pred, axis=1)

# collapse multi-class preds into binary preds
# preds_detection = np.sum(mask_pred, axis=1)

pred_tumor_binary = None
pred_immune_binary = None
pred_binary = None

for i in range(mask_pred.shape[0]):

    tumor_mask = mask_pred[i,0,:,:]!=0
    immune_mask = mask_pred[i,1,:,:]!=0
    mask = preds_detection[i,:,:]!=0

    preds_tumor_mask = tumor_mask[np.newaxis, :]
    preds_immune_mask = immune_mask[np.newaxis,:]
    preds_mask = mask[np.newaxis,:]

    if i == 0:
        pred_tumor_binary = preds_tumor_mask
        pred_immune_binary = preds_immune_mask
        pred_binary = preds_mask
    else:
        pred_tumor_binary = np.concatenate([pred_tumor_binary,preds_tumor_mask],axis = 0)
        pred_immune_binary = np.concatenate([pred_immune_binary,preds_immune_mask],axis = 0)
        pred_binary = np.concatenate([pred_binary,preds_mask],axis = 0)

pred_tumor_binary = pred_tumor_binary[np.newaxis, :]
pred_immune_binary = pred_immune_binary[np.newaxis,:]
pred_binary = pred_binary[np.newaxis,:]

use_weight = 1
loss_weight = rp.get_loss_weight(256, 256, use_weight,border=16)

preds_tumor_test = np.transpose(pred_tumor_binary,(1,0,2,3))
preds_immune_test = np.transpose(pred_immune_binary,(1,0,2,3))
preds_test = np.transpose(pred_binary,(1,0,2,3))

preds_tumor_test = preds_tumor_test[0:patches_imgs_test.shape[0],:,:,:]
preds_immune_test = preds_immune_test[0:patches_imgs_test.shape[0],:,:,:]
preds_test = preds_test[0:patches_imgs_test.shape[0],:,:,:]

pred_tumor_img = rp.recompose_overlap(preds_tumor_test,block_height,block_weight,128,128,1,loss_weight=loss_weight)
pred_immune_img = rp.recompose_overlap(preds_immune_test,block_height,block_weight,128,128,1,loss_weight=loss_weight)
pred_img = rp.recompose_overlap(preds_test,block_height,block_weight,128,128,1,loss_weight=loss_weight)

# pred_img = (pred_img > 0.5).astype(np.uint8)
pred_tumor_img = pred_tumor_img[:,:,0:block_height,0:block_weight]
pred_immune_img = pred_immune_img[:,:,0:block_height,0:block_weight]
pred_img = pred_img[:,:,0:block_height,0:block_weight]

tumor_test_result = './block_result/model_result/mask_tumor/'
immune_test_result = './block_result/model_result/mask_immune/'
test_result = './block_result/model_result/mask/'

#save the result

for i in range(pred_tumor_img.shape[0]):
    rp.visualize(np.transpose(pred_tumor_img[i,:,:,:], (1,2,0)), tumor_test_result + im_names[i])
    rp.visualize(np.transpose(pred_immune_img[i, :, :, :], (1, 2, 0)), immune_test_result + im_names[i])
    rp.visualize(np.transpose(pred_img[i, :, :, :], (1, 2, 0)), test_result + im_names[i])
end = time.time()
print('Running time: %s Seconds' % (end - start))