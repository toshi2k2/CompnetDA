# Official Implementation of "A Bayesian Approach to OOD Robustness in Image Classification" [CVPR-2024]
 	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-bayesian-approach-to-ood-robustness-in/unsupervised-domain-adaptation-on-ood-cv)](https://paperswithcode.com/sota/unsupervised-domain-adaptation-on-ood-cv?p=a-bayesian-approach-to-ood-robustness-in)
 
### Installation

The code has been tested on **Python 3.9** and PyTorch GPU version 1.10.1, with CUDA-11.3.

#### Quick Start

```
cd CompDA
```

Put data in data/OODCV folder and create a 'models' folder-

```
mkdir data && mkdir models && mkdir results && mkdir baseline_models
```

First train/finetune a VGG16(BN) model on OOD-CV(called ROBIN earlier) training data (to be used as backbone) and save it as 

```
baseline_models/Robin-train-vgg_bn.pth
```

To run and evaluate experiment for VGG16 on OOD-CV dataset, run the following-

```
python Initialization_Code/vmf_cluster.py --da True --robin_cat None<0,1,2,3,4 for OOD-CV subcategories> && python Initialization_Code/simmat.py --da True --mode 'mixed' && python Initialization_Code/Learn_mix_model_vMF_view.py --da True --mode 'mixed' && python Code/test.py --da True --bbone 'vgg_tr' --sveimglst True --load False && python Da/pseudo_data_creation.py
```

Change dataset variable in Code/config.py and Initialization_Code/config_initialization.py from 'robin' to 'pseudorobin' and run-

```
python Initialization_Code/simmat.py --da True --mode '' --dataset pseudorobin && python Initialization_Code/Learn_mix_model_vMF_view.py --da True --mode '' --dataset pseudorobin && python Code/train.py --gce True --dataset pseudorobin
```

A folder with the format models/vcvgg_tr_final/vc_{epoch-number}.pth will be created after running the train.py file. Change lines 179-181 in Code/test.py according to the model name created, and run evaluation (for Combined nuisance)-

```
python Code/test.py
```

<!-- #### Evaluate the occluder localization performance of a model

If you want to test occluder localization run:
```
python Code/eval_occlusion_localization.py
``` 
This will output qualitative occlusion localization results for each image and a quantitative analysis over all images 
as ROC curve.

## Initializing CompositionalNet Parameters

We initialize CompositionalNets (i.e. the vMF kernels and mixture models) by clustering the training data. 
In particular, we initialize the vMF kernels by clustering the feature vectors:

```
python Initialization_Code/vMF_clustering.py
``` 

Furthermore, we initialize the mixture models by EM-type learning.
The initial cluster assignment for the EM-type learning is computed based on the similarity of the vMF encodings of the training images.
To compute the similarity matrices use:
 
```
python Initialization_Code/comptSimMat.py
``` 

As this process takes some time we provide precomputed similarity matrices [here](https://livejohnshopkins-my.sharepoint.com/:u:/g/personal/akortyl1_jh_edu/EU6OcwaW7l1IhpggHJBCjeIBB_xLd28bDUIcoPHKUOhxqg?e=5k34Nx), you need to copy them into the 'models/init_vgg/' folder.
Afterwards you can compute the initialization of the mixture models by executing:

```
python Initialization_Code/Learn_mix_model_vMF_view.py
```

## Relation to Prior Work

This work (and code) is based on the following work -
```
Compositional Convolutional Neural Networks: A Deep Architecture with Innate Robustness to Partial Occlusion
Adam Kortylewski, Ju He, Qing Liu, Alan Yuille
CVPR 2020
``` -->

<!-- ## Contact

If you have any questions you can contact Adam Kortylewski. -->

<!-- ## Acknowledgement

We thank Zhishuai Zhang for helping us speed up and clean the code for the release. -->
