import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import pandas as pd
from torch.optim.lr_scheduler import StepLR
import albumentations as A
from utils.ihc_data import PanNukeDataModule
from utils.SRSANet import SRSANet, loss_hovernet, post_process_batch_hovernet
from pathml.ml.utils import wrap_transform_multichannel, dice_score
from pathml.utils import plot_segmentation,segmentation_lines

n_classes_pannuke = 3

#training with multi GPUs
print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda:0")
print(f"Device:\t\t{device}")


# load the model
SRSAnet = SRSANet(n_classes=n_classes_pannuke)

# wrap model to use multi-GPU
srsanet = torch.nn.DataParallel(SRSAnet)

# data augmentation transform
srsa_transform = A.Compose(
    [A.VerticalFlip(p=0.5),
     A.HorizontalFlip(p=0.5),
     A.RandomRotate90(p=0.5),
     A.GaussianBlur(p=0.5),
     A.MedianBlur(p=0.5, blur_limit=5)],
    additional_targets = {f"mask{i}" : "mask" for i in range(n_classes_pannuke)}
)

transform = wrap_transform_multichannel(srsa_transform)

#load pannukedata
ihc_data = PanNukeDataModule(
    data_dir="./dataset/",
    download=False,
    nucleus_type_labels=True,
    batch_size=16,
    hovernet_preprocess=True,
    split=1,
    transforms=transform
)

train_dataloader = ihc_data.train_dataloader
valid_dataloader = ihc_data.valid_dataloader
test_dataloader = ihc_data.test_dataloader
images, masks, hvs, types,stems = next(iter(train_dataloader))

# set up optimizer
opt = torch.optim.Adam(srsanet.parameters(), lr = 1e-4)
# learning rate scheduler to reduce LR by factor of 10 each 25 epochs
scheduler = StepLR(opt, step_size=25, gamma=0.1)

# send model to GPU
srsanet.to(device)

# load the best model
checkpoint = torch.load("./self-built-dataset-checkpoint/srsanet_best_perf.pt")

srsanet.load_state_dict(checkpoint)
srsanet.eval()

ims = None
mask_truth = None
mask_pred = None
tissue_types = []
stems_list = []
with torch.no_grad():
    for i, data in tqdm(enumerate(test_dataloader)):
        # send the data to the GPU
        images = data[0].float().to(device)
        masks = data[1].to(device)
        hv = data[2].float().to(device)
        tissue_type = data[3]
        stems = data[4]
        for j in stems:
            stems_list.append(j)

        # pass thru network to get predictions
        outputs = srsanet(images)
        preds_detection, preds_classification = post_process_batch_hovernet(outputs, n_classes=3)

        if i == 0:
            ims = data[0].numpy()
            mask_truth = data[1].numpy()
            mask_pred = preds_classification
            tissue_types.extend(tissue_type)
        else:
            ims = np.concatenate([ims, data[0].numpy()], axis=0)
            mask_truth = np.concatenate([mask_truth, data[1].numpy()], axis=0)
            mask_pred = np.concatenate([mask_pred, preds_classification], axis=0)
            tissue_types.extend(tissue_type)
# collapse multi-class preds into binary preds
preds_detection = np.sum(mask_pred, axis=1)
dice_scores = np.empty(shape = len(tissue_types))


# change image tensor from (B, C, H, W) to (B, H, W, C)
# matplotlib likes channels in last dimension

n = 35
for i in range(n):
    mask_tumor = mask_pred[i,0,:,:]!=0
    mask_immune = mask_pred[i,1,:,:]!=0
    mask = mask_pred[i,2,:,:]!=0
    multi_mask=mask_pred[i,...]!=0
    # true_tumor = mask_truth[i,0,:,:]!=0
    # true_immune = mask_truth[i,1,:,:]!=0
    # true = mask_truth[i,2,:,:]!=0
    plt.imsave('./patch-result/mask-tumor/{}.png'.format(stems_list[i]),mask_tumor,cmap = 'gray')
    plt.imsave('./patch-result/mask-immune/{}.png'.format(stems_list[i]), mask_immune,cmap = 'gray')
    plt.imsave('./patch-result/2-channel/{}.png'.format(stems_list[i]),mask,cmap = 'gray')
    np.save('./patch-result/multi-mask/{}.npy'.format(stems_list[i]), multi_mask)
    # plt.imsave('./true-tumor/{}.png'.format(stems[i]),true_tumor)
    # plt.imsave('./true-immune/{}.png'.format(stems[i]), true_immune)
    # plt.imsave('./true/{}.png'.format(stems[i]),true)

for i in range(n):
    plt.imsave('./patch-result/np-pre-result/{}.png'.format(stems_list[i]),preds_detection[i,...]!=0,cmap = 'gray')
