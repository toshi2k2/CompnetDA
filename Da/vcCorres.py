"""Building correspondence between VCs"""
import os, sys
from numpy.core.fromnumeric import resize, shape

from torchvision import models
p = os.path.abspath('.')
sys.path.insert(1, p)
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from Initialization_Code.vcdist_funcs import vc_dis_paral, vc_dis_paral_full
import time
import pickle
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, \
    cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer, init_path, \
    dict_dir, sim_dir, extractor, da_sim_dir, da_dict_dir
from Code.helpers import getImg, imgLoader, Imgset, myresize
from torch.utils.data import DataLoader
import numpy as np
# import math
# import torch
import matplotlib.pyplot as plt
import cv2

mode = '1-1' #? [simple, cutoff, 1-1]
stat = False
# dict_dir = 'models_clean/init_vgg/dictionary_vgg/'
# da_dict_dir = 'models/da_init_vgg_bn/dictionary_vgg_bn/'
dict_dir = 'models/init_vgg_bn/dictionary_vgg_bn/'
da_dict_dir = 'models_robin_all/init_vgg_bn/dictionary_vgg_bn/'
# dict_dir = 'models/init_vgg_bn/dictionary_vgg_bn/'
# dict_dir = 'models_robin_all/init_vgg_bn/dictionary_vgg_bn/'
print(dict_dir)

#/ Load Vmf kernels learned by kmeans++ clustering
print("Loading corrupted Vmf Kernels")
with open(da_dict_dir+'dictionary_{}_{}.pickle'.format(layer, vc_num), 'rb') as fh:
    centers_cor = pickle.load(fh)
print("Loading clean Vmf Kernels")
with open(dict_dir+'dictionary_{}_{}.pickle'.format(layer, vc_num), 'rb') as fh:
    centers_cln = pickle.load(fh)
# with open(dict_dir+'dictionary_{}.pickle'.format(layer), 'rb') as fh:
#     centers_cln = pickle.load(fh)

print(centers_cln.shape, centers_cor.shape)

r_set = cdist(centers_cln, centers_cor, 'cosine')
# fig, ax = plt.subplots()
# cax = fig.add_axes([0.27, 0.95, 0.5, 0.05])
# im = ax.imshow(r_set)
# fig.colorbar(im, cax=cax, orientation='horizontal')
# plt.show()

#/ Statistics ##
if stat:
    for i in range(1,10):
        p = 1-i*0.1
        print(p, ">: ", r_set[r_set<p].sum()/r_set.size * 100,"%")
    plt.hist(r_set)
    plt.show()
    # plt.savefig()
# exit()

mns, midx = r_set.min(1), r_set.argmin(1)

new_center = centers_cln
if mode in ['simple', 'all']:
    for i, x in enumerate(midx):
        new_center[i] = centers_cor[x]

if mode in ['cutoff', 'all']:
    print(mode)
    cut = 0.2
    inx = 0
    for i, x in enumerate(midx):
        if mns[i] <= cut:
            new_center[i] = centers_cor[x]
            inx+=1
        # else:
        elif mns[i] > 0.5:
            new_center[i] = new_center[i]*0.+ 1e-9
    print("Preserved VCs: ", 100.*(inx/512),"%", inx)

if mode == '1-1':
    """1-to-1 VC mapping. VCs can't be reused even if they are close to multiple VCs in the other domain"""
    # unique, counts = np.unique(midx, return_counts=True)
    # for x,y in zip(unique, counts): print(x,":",y)
    tmp = r_set.copy()
    min_list = [None]*r_set.shape[0]
    ix = 0
    for i in range(r_set.shape[0]):
        u, v = divmod(tmp.argmin(), tmp.shape[1])
        min_list[u]=v
        new_center[u] = centers_cor[v]
        if r_set[u,v] <=0.5:
            ix+=1
        tmp[u,:]=1.1
        tmp[:,v]=1.1
    # checks
    assert(not None in min_list)
    print(ix/r_set.shape[0] * 100. ,"%")
    # unique, counts = np.unique(np.array(min_list), return_counts=True)
    # for x,y in zip(unique, counts): print(x,":",y)

# with open(da_dict_dir+'corres_dict_{}_{}.pickle'.format(layer, vc_num), 'wb') as fh:
#     pickle.dump(new_center, fh)

# d_pth = dict_dir+'/cluster_images_pool5_512/'
# c_pth = da_dict_dir+'/cluster_images_pool5_512/'
# for i, x in enumerate(midx):
#     img1 = cv2.imread(d_pth+str(i)+'.JPEG')
#     img2 = cv2.imread(c_pth+str(x)+'.JPEG')
#     patch = np.concatenate((img1, img2), axis=1)

#     cv2.imwrite('./res_sim_corres/'+str(round(mns[i],3))+'_'+str(i)+'.JPEG',patch)
