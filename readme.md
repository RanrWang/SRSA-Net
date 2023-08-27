# 'Simultaneously segmenting and classifying cell nuclei by using multi-task learning in multiplex immunohistochemical tissue microarray sections'
## This folder contains the implementation of the codes.

## Requirements: 
The details of environment such as python package can reference 'environment.txt'.

As for pathml(python package), there is one bug.

If you encounter this error:
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
in pathml.ml.utils dice loss,

line 73 true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]

line 82 true_1_hot = torch.eye(num_classes)[true.squeeze(1)]

please modify as:
line 73 true_1_hot = torch.eye(num_classes + 1, device=true.device)[true.squeeze(1)]

line82 true_1_hot = torch.eye(num_classes, device=true.device)[true.squeeze(1)]
## Data:

It can be downloaded at: https://zenodo.org/record/7647846.

IHC Cohort: Multiplex IHC stained histological slides were collected from Liaoning Cancer Hospital and Institute in China. The raw data collected are TMA sections, all of which are obtained from NSCLC patients. Non-overlapping image patches (256*256 pixels) are extracted from TMA sections and manually annotated by our collaborated pathologists via the Qupath software. Initially, seven TMA sections with multiplex stains including CD3, CD20, CD38, CDK4, Cyclin-D1, Ki67, and P53 were used. Except for CD3 stained TMA sections, we randomly cropped 25 image patches from each of these TMA sections, of which 17, 3 and 5 patches are correspondingly used for training, validation, and testing. Note that 18, 3 and 5 CD3 image patches are cropped for building dataset. To suppress over-fitting and enhance generalization of the SRSA-Net, we additionally included 81 annotated image patches from other five TMA sections with stains of CD34, CD68, D2-40, FAP, and SMA into training and validation set. In total, the IHC cohort includes 9,725 manually identified cell nuclei. The numbers of training, validation and testing image patches are 195, 36, and 35, respectively.


ihc data

   dataset
   ├── images
   │   ├── fold1_0_CDK4.png
   │   ├── fold1_1_D2.png
   │   ├── fold1_2_D2.png
   │   └── ...
   │
   └── masks
       ├── fold1_0_CDK4.npy
       ├── fold1_1_D2.npy
       ├── fold1_2_D2.npy
       └── ...

For the masks, the first channel and the second channel denotes the negative and positive
nuclei pixels, respectively. Note that each pixel is labelled from 0 to n, where n is the
number of individual nuclei detected. 0 pixels indicate background. Pixel values i indicate that the pixel belongs to the ith nucleus.  
The last channel marks the information of all the nuclei pixels, where nuclei pixel are left as 0, otherwise 1.

## Usage of Directories
dataset: store data.

glog-post-patch: process glog-based algorithm. main file: glog_multimask.py.

utils: tools and models.

## Start to train
1.set up environment.

2.configure the path of dataset and checkpoint in 'train.py'.

3.run 'train.py'.

## Test model
1.run 'save-test.py' to generate prediction results. 

2.calculate related evaluation metircs.

## Process a large-scale pathology image such as tissue core.
1.configure path and hyper-parameters in 'quick-block-test.py' such as stride.

2.run 'quick-block-test.py'

## Citation

@article{rosenthal2022building,
  title={Building tools for machine learning and artificial intelligence in cancer research: best practices and a case study with the PathML toolkit for computational pathology},
  author={Rosenthal, Jacob and Carelli, Ryan and Omar, Mohamed and Brundage, David and Halbert, Ella and Nyman, Jackson and Hari, Surya N and Van Allen, Eliezer M and Marchionni, Luigi and Umeton, Renato and others},
  journal={Molecular Cancer Research},
  volume={20},
  number={2},
  pages={202--206},
  year={2022},
  publisher={AACR}
}

@article{graham2019hover,
  title={Hover-net: Simultaneous segmentation and classification of nuclei in multi-tissue histology images},
  author={Graham, Simon and Vu, Quoc Dang and Raza, Shan E Ahmed and Azam, Ayesha and Tsang, Yee Wah and Kwak, Jin Tae and Rajpoot, Nasir},
  journal={Medical Image Analysis},
  pages={101563},
  year={2019},
  publisher={Elsevier}
}

@inproceedings{gamper2019pannuke,
  title={PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification},
  author={Gamper, Jevgenij and Koohbanani, Navid Alemi and Benet, Ksenija and Khuram, Ali and Rajpoot, Nasir},
  booktitle={European Congress on Digital Pathology},
  pages={11--19},
  year={2019},
  organization={Springer}
}
