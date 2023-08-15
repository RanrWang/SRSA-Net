'''To train the SRSA-Net to simultaneously segment and classify nuclei.
 the related codes reference Hover-Net and python package 'pathml'.
 'Hover-Net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images'.
 '''
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
import os
from torch.optim.lr_scheduler import StepLR
import albumentations as A
from utils.ihc_data import PanNukeDataModule
from utils.SRSANet import SRSANet, loss_hovernet, post_process_batch_hovernet
from pathml.ml.utils import wrap_transform_multichannel, dice_score
from pathml.utils import plot_segmentation
import time
start = time.time()
#set class number
n_classes_pannuke = 3

# load the model
SRSAnet = SRSANet(n_classes=n_classes_pannuke)

# wrap model to use multi-GPU
srsanet = torch.nn.DataParallel(SRSAnet)
#training with multi GPUs
print(f"GPUs used:\t{torch.cuda.device_count()}")
device = torch.device("cuda:0")
print(f"Device:\t\t{device}")
print(torch.cuda.current_device())
# net_dict = srsanet.state_dict()
# predict_model = torch.load('resnet50.pth')
# state_dict = {k: v for k,v in predict_model.items()
#                  if k in net_dict.keys()}
# net_dict.update(state_dict)
# srsanet.load_state_dict(net_dict)

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
    batch_size=8,
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


##train
n_epochs = 100

# print performance metrics every n epochs
print_every_n_epochs = 5

# evaluating performance on a random subset of validation mini-batches
# this saves time instead of evaluating on the entire validation set
n_minibatch_valid =5

epoch_train_losses = {}
epoch_valid_losses = {}
epoch_train_dice = {}
epoch_valid_dice = {}

best_epoch = 0

# main training loop
for i in tqdm(range(n_epochs)):
    minibatch_train_losses = []
    minibatch_train_dice = []

    # put model in training mode
    srsanet.train()

    for data in train_dataloader:
        # send the data to the GPU
        images = data[0].float().to(device)
        masks = data[1].to(device)
        hv = data[2].float().to(device)
        tissue_type = data[3]
        stems = data[4]

        # zero out gradient
        opt.zero_grad()

        # forward pass
        outputs = srsanet(images)

        # compute loss
        loss = loss_hovernet(outputs = outputs, ground_truth = [masks, hv], n_classes=3)

        # track loss
        minibatch_train_losses.append(loss.item())

        # also track dice score to measure performance
        preds_detection, preds_classification = post_process_batch_hovernet(outputs, n_classes=n_classes_pannuke)
        truth_binary = masks[:, -1, :, :] == 0
        dice = dice_score(preds_detection, truth_binary.cpu().numpy())
        minibatch_train_dice.append(dice)

        # compute gradients
        loss.backward()

        # step optimizer and scheduler
        opt.step()

    #step LR scheduler
    scheduler.step()

    # evaluate on random subset of validation data
    srsanet.eval()
    minibatch_valid_losses = []
    minibatch_valid_dice = []
    # randomly choose minibatches for evaluating
    minibatch_ix = np.random.choice(range(len(valid_dataloader)), replace=False, size=n_minibatch_valid)
    with torch.no_grad():
        for j, data in enumerate(valid_dataloader):
            if j in minibatch_ix:
                # send the data to the GPU
                images = data[0].float().to(device)
                masks = data[1].to(device)
                hv = data[2].float().to(device)
                tissue_type = data[3]

                # forward pass
                outputs = srsanet(images)

                # compute loss
                loss = loss_hovernet(outputs = outputs, ground_truth = [masks, hv], n_classes=3)

                # track loss
                minibatch_valid_losses.append(loss.item())

                # also track dice score to measure performance
                preds_detection, preds_classification = post_process_batch_hovernet(outputs, n_classes=n_classes_pannuke)
                truth_binary = masks[:, -1, :, :] == 0
                dice = dice_score(preds_detection, truth_binary.cpu().numpy())
                minibatch_valid_dice.append(dice)

    # average performance metrics over minibatches
    mean_train_loss = np.mean(minibatch_train_losses)
    mean_valid_loss = np.mean(minibatch_valid_losses)
    mean_train_dice = np.mean(minibatch_train_dice)
    mean_valid_dice = np.mean(minibatch_valid_dice)

    # save the model with best performance
    if i != 0:
        if mean_valid_loss < min(epoch_valid_losses.values()):
            best_epoch = i
            torch.save(srsanet.state_dict(), f"srsanet_best_perf.pt")

    # track performance over training epochs
    epoch_train_losses.update({i : mean_train_loss})
    epoch_valid_losses.update({i : mean_valid_loss})
    epoch_train_dice.update({i : mean_train_dice})
    epoch_valid_dice.update({i : mean_valid_dice})

    if print_every_n_epochs is not None:
        if i % print_every_n_epochs == print_every_n_epochs - 1:
            print(f"Epoch {i+1}/{n_epochs}:")
            print(f"\ttraining loss: {np.round(mean_train_loss, 4)}\tvalidation loss: {np.round(mean_valid_loss, 4)}")
            print(f"\ttraining dice: {np.round(mean_train_dice, 4)}\tvalidation dice: {np.round(mean_valid_dice, 4)}")

# save fully trained model
torch.save(srsanet.state_dict(), f"srsanet_fully_trained.pt")
print(f"\nEpoch with best validation performance: {best_epoch}")

fix, ax = plt.subplots(nrows=1, ncols=2, figsize = (10, 4))

ax[0].plot(epoch_train_losses.keys(), epoch_train_losses.values(), label = "Train")
ax[0].plot(epoch_valid_losses.keys(), epoch_valid_losses.values(), label = "Validation")
ax[0].scatter(x=best_epoch, y=epoch_valid_losses[best_epoch], label = "Best Model",
              color = "green", marker="*")
ax[0].set_title("Training: Loss")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()

ax[1].plot(epoch_train_dice.keys(), epoch_train_dice.values(), label = "Train")
ax[1].plot(epoch_valid_dice.keys(), epoch_valid_dice.values(), label = "Validation")
ax[1].scatter(x=best_epoch, y=epoch_valid_dice[best_epoch], label = "Best Model",
              color = "green", marker="*")
ax[1].set_title("Training: Dice Score")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Dice Score")
ax[1].legend()
# plt.show()
end = time.time()
print('Running time: %s Seconds' % (end - start))