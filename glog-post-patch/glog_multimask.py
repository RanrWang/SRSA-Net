'''It's essential to properly set up the packages, particularly 'matlab'..
the matlab codes are written by Hongming Xu.
reference:'Automatic Nuclei Detection Based on Generalized Laplacian of Gaussian Filters'. Hongming Xu, Cheng Lu...'''
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from skimage.morphology import remove_small_objects
import cv2
import os
import matlab
import matlab.engine
eng = matlab.engine.start_matlab()
def selected(image_folder,mask_folder,save_folder):
    mask_list = os.listdir(mask_folder)
    for mask_name in mask_list:
        mask_path = os.path.join(mask_folder,mask_name)
        im_name=mask_name.replace('.npy','.png')
        im_path = os.path.join(image_folder,im_name)
        save_path = os.path.join(save_folder,mask_name)
        mask = np.load(mask_path)
        # mask = cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
        im = cv2.imread(im_path,0)
        multi_mask = np.zeros((mask.shape))
        for i in range(mask.shape[0]):
            labels = measure.label(mask[i,:,:])
            number = list(np.unique(labels))
            number.remove(0)
            lar_mask = np.zeros((256,256))
            small_mask = np.zeros((256,256))
            midd_mask = np.zeros((256,256))
            for n in number:
                region = labels == n
                region = np.array(region,np.int)
                if np.sum(region)>1300:
                    lar_mask += region*255
                elif np.sum(region)<1300 and np.sum(region)>1100:
                    midd_mask += region*255
                else:
                    small_mask += region*255
            if np.sum(lar_mask)==0:
                glog_lar_mask = lar_mask
            else:
                glog_lar_mask = eng.np_glog_process(matlab.double(im.tolist()),
                                                    matlab.double(lar_mask.tolist()),
                                                    matlab.double([18]),
                                                    matlab.double([14]),
                                                    matlab.double([15]))
            if np.sum(midd_mask)==0:
                glog_midd_mask = midd_mask
            else:
                # glog_mask = lar_mask
                glog_midd_mask = eng.np_glog_process(matlab.double(im.tolist()),
                                                matlab.double(midd_mask.tolist()),
                                                matlab.double([15]),
                                                matlab.double([12]),
                                                matlab.double([11]))
            final_mask = glog_lar_mask + np.array(glog_midd_mask,dtype=int) + small_mask
            # final_mask = glog_lar_mask + small_mask
            final_mask[final_mask>1]=1
            final_mask = remove_small_objects(np.array(final_mask, dtype=bool), min_size=80)
            final_mask = np.array((final_mask),dtype=np.int)
            final_mask *= 255
            multi_mask[i,:,:]=final_mask
        np.save('{}'.format(save_path),multi_mask)
image_folder = './self-built/imgs'
mask_folder = './SRSA-multi-mask'
save_folder = './self-built/glog-multi-mask'

selected(image_folder,mask_folder,save_folder)


