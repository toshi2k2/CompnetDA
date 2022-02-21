# To calculate compsimmat with bigger spatial VCs

import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist
from vcdist_funcs import vc_dis_paral, vc_dis_paral_full
import time
import pickle
from config_initialization import vc_num, dataset, categories, data_path, \
    cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer, init_path, \
    dict_dir, sim_dir, extractor, da_sim_dir, da_dict_dir, robin_cats
from Code.helpers import getImg, imgLoader, Imgset, myresize
from torch.utils.data import DataLoader
import numpy as np
import math, random
import torch

DA = True
corr = None#'snow'  # 'snow'
mode = 'mixed'#'corres'  # * mixed: vc- corrupted, mixture - clean; '': vc and mix - corrupted;reverse:
           # * reverse: vc-clean, mix-corrupt; None, 'corres': correspondence dict b/w clean and corr VCs
vc_space = 0#3
ignore_edge = False#True # ignore edge of layer output - to reduce comptuations
add_data = False # add data to current data

paral_num = 10
nimg_per_cat = 5000
imgs_par_cat = np.zeros(len(categories))
occ_level = 'ZERO'
occ_type = ''

print('{} max_images {}'.format(dataset, nimg_per_cat))

if dataset in ['robin','pseudorobin']:
    categories.remove('bottle')
    cat_test = categories
    # cat = [robin_cats[0]]
    cat = None

if DA or mode in ['mixed', '', 'reverse']:
    if not os.path.exists(da_sim_dir):
        print("creating {}".format(da_sim_dir))
        os.makedirs(da_sim_dir)
else:
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)

#############################
# BEWARE THIS IS RESET TO LOAD OLD VCS AND MODEL
#############################
#/ Load Vmf kernels learned by kmeans++ clustering
if DA and mode in ['mixed', '']:
    print("Loading corrupted/D2 Vmf Kernels")
    with open(da_dict_dir+'dictionary_{}_{}.pickle'.format(layer, vc_num), 'rb') as fh:
        centers = pickle.load(fh)
elif DA and mode in ['corres']:
    print("Loading CORRESPONDENCE Vmf Kernels")
    with open(da_dict_dir+'corres_dict_{}_{}.pickle'.format(layer, vc_num), 'rb') as fh:
        centers = pickle.load(fh)
else:
    print("Loading clean/D1 Vmf Kernels")
    with open(dict_dir+'dictionary_{}_{}.pickle'.format(layer, vc_num), 'rb') as fh:
        centers = pickle.load(fh)
##HERE
bool_pytorch = True

for category in categories:  # * loading individual class categories
    cat_idx = categories.index(category)
    print('Category {} - {} / {}'.format(category, cat_idx, len(categories)))
    if DA and mode not in ['mixed', 'corres']:
        if dataset in ['robin','pseudorobin']:
            print("Loading Robin test data (pseudo {})".format(dataset=='pseudorobin'))
            imgs, labels, masks = getImg('test', cat_test, dataset, data_path, [category], \
                occ_level, occ_type, bool_load_occ_mask=False, subcat=cat) 
        else:
            print("Loading corrupted data")
            imgs, labels, masks = getImg('train', [category], dataset, data_path, cat_test,\
                occ_level, occ_type, bool_load_occ_mask=False, determinate=True, corruption=corr)
    else:
        print("Loading clean\D1 data\n")
        imgs, labels, masks = getImg('train', [category], dataset, data_path, cat_test, \
            occ_level, occ_type, bool_load_occ_mask=False)
    imgs = imgs[:nimg_per_cat]
    if add_data:
        print("Adding clean data from robin train!")
        frc = 0.5
        imgs2, labels2, masks2 = getImg('train', [category], 'robin', data_path, cat_test, \
            occ_level, occ_type, bool_load_occ_mask=False)
        imgs+=random.sample(imgs2, int(min(len(imgs), len(imgs2))*frc))
        labels+=labels2[:int(min(len(labels), len(labels2))*frc)]
        masks+=masks2[:int(min(len(masks), len(masks2))*frc)]
    N = len(imgs)
    ## HERE
    # imgset = Imgset(imgs, masks, labels, imgLoader, bool_square_images=False,bool_cutout=False,\
    # 	bool_pytorch=bool_pytorch)  # ! extra parameters!
    imgset = Imgset(imgs, masks, labels, imgLoader, bool_square_images=False)
    data_loader = DataLoader(dataset=imgset, batch_size=1, shuffle=False)
    if DA or mode in ['mixed', '', 'corres']:
        savename = os.path.join(da_sim_dir, 'simmat_mthrh045_{}_K{}.pickle'.format(category,vc_num))
        print("savename ", savename)
    else:
        savename = os.path.join(sim_dir, 'simmat_mthrh045_{}_K{}.pickle'.format(category,vc_num))

    if not os.path.exists(savename):
    # if True:
        r_set = [None for nn in range(N)]  # * len - 986
        for ii, data in enumerate(data_loader):
            input, mask, label = data
            if imgs_par_cat[cat_idx]<N:
                with torch.no_grad():
                    layer_feature = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()  # * Feature extration - [512, 7, 22]
                iheight, iwidth = layer_feature.shape[1:3]  # [7, 22]
                if vc_space in [2,3]:
                    if vc_space==3: ds = 9
                    elif vc_space==2: ds = 3
                    else: raise(RuntimeError)
                    # layer_feature = np.pad(layer_feature)
                    # new_arr = np.zeros((layer_feature.shape[0]*ds, iheight, iwidth))
                        # new_arr = np.zeros((layer_feature.shape[0]*ds, layer_feature.shape[1], layer_feature.shape[2]))
                    if not ignore_edge:
                        new_arr = np.zeros((layer_feature.shape[0]*ds, iheight, iwidth))
                        x_ = np.pad(layer_feature, ((0, 0), (1, 1), (1, 1)), mode='edge')
                        ic, jc = x_.shape[1]-1, x_.shape[2]-1
                    else:
                        new_arr = np.zeros((layer_feature.shape[0]*ds, iheight-2, iwidth-2))
                        ic, jc = iheight-1, iwidth-1
                    for i in np.arange(1,ic):
                        for j in np.arange(1,jc):
                            if vc_space == 3:
                                if not ignore_edge:
                                    t = x_[:,i-1:i+2,j-1:j+2]
                                else:
                                    t = layer_feature[:,i-1:i+2,j-1:j+2]
                            # elif vc_space == 2:
                            #     t = x_[:,i-1:i+2,:]
                            t = t.reshape(t.shape[0]*ds, -1)
                            new_arr[:,i-1,j-1]=t[:,0]
                    lff = new_arr.reshape(new_arr.shape[0], -1).T
                    del new_arr 
                else:
                    lff = layer_feature.reshape(layer_feature.shape[0], -1).T  # * 2D to 1D per feature vector [154,512]
                lff_norm = lff / (np.sqrt(np.sum(lff ** 2, 1) + 1e-10).reshape(-1, 1)) + 1e-10
                #/ Calculate cosine distance between image and all vmf kernels/means/centers
                if not ignore_edge:
                    r_set[ii] = cdist(lff_norm, centers, 'cosine').reshape(iheight, iwidth, -1)   # / [7, 22, 512] - 512 values for each pixel
                else:
                    r_set[ii] = cdist(lff_norm, centers, 'cosine').reshape(iheight-2, iwidth-2, -1)
                imgs_par_cat[cat_idx] += 1

        print('Determine best threshold for binarization - {} ...'.format(category))
        nthresh = 20
        magic_thhs = range(nthresh)
        coverage = np.zeros(nthresh)
        act_per_pix = np.zeros(nthresh)
        layer_feature_b = [None for nn in range(100)]
        magic_thhs = np.asarray([x*1/nthresh for x in range(nthresh)])  # * array of 20 equidistant values between 0 and 1

        if len(r_set)>=100:
            range_nums = 100
        else:
            range_nums = len(r_set)
            print("No. of images is <100: {}".format(range_nums))
        for idx, magic_thh in enumerate(magic_thhs):
            for nn in range(range_nums):  # ! using only 100/986 images?
                layer_feature_b[nn] = (r_set[nn] < magic_thh).astype(int).T  # * [512, 22, 7] mask
                coverage[idx] += np.mean(np.sum(layer_feature_b[nn], axis=0)>0) #/ ??
                act_per_pix[idx] += np.mean(np.sum(layer_feature_b[nn], axis=0)) #/ Avg. number of vc activations/pixel
        coverage = coverage/100
        act_per_pix = act_per_pix/100
        best_loc = (act_per_pix > 2) * (act_per_pix < 15)  # / mask for location? - at least two centers activated?

        if np.sum(best_loc):
            best_thresh = np.min(magic_thhs[best_loc])
        else:
            best_thresh = 0.45
        layer_feature_b = [None for nn in range(N)]
        for nn in range(N):  # / using all images here?
            layer_feature_b[nn] = (r_set[nn] < best_thresh).astype(int).T

        print('Start compute sim matrix ... magicThresh {}'.format(best_thresh))
        _s = time.time()

        mat_dis1 = np.ones((N, N))
        mat_dis2 = np.ones((N, N))
        N_sub = 200  # / what's this?
        sub_cnt = int(math.ceil(N/N_sub))  # 5
        for ss1 in range(sub_cnt):
            start1 = ss1*N_sub
            end1 = min((ss1+1)*N_sub, N)
            layer_feature_b_ss1 = layer_feature_b[start1:end1]
            for ss2 in range(ss1, sub_cnt):
                print('iter {1}/{0} {2}/{0}'.format(sub_cnt, ss1+1, ss2+1))
                _ss = time.time()
                start2 = ss2*N_sub
                end2 = min((ss2+1)*N_sub, N)
                if ss1 == ss2:
                    inputs = [(layer_feature_b_ss1, nn) for nn in range(end2-start2)]
                    para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral)(i) for i in inputs)) #! ?

                else:
                    layer_feature_b_ss2 = layer_feature_b[start2:end2]
                    inputs = [(layer_feature_b_ss2, lfb) for lfb in layer_feature_b_ss1]
                    para_rst = np.array(Parallel(n_jobs=paral_num)(delayed(vc_dis_paral_full)(i) for i in inputs))

                mat_dis1[start1:end1, start2:end2] = para_rst[:,0]
                mat_dis2[start1:end1, start2:end2] = para_rst[:,1]

                _ee = time.time()
                print('comptSimMat iter time: {}'.format((_ee-_ss)/60))

        _e = time.time()
        print('comptSimMat total time: {}'.format((_e-_s)/60))

        with open(savename, 'wb') as fh:
            print('saving at: '+savename)
            pickle.dump([mat_dis1, mat_dis2], fh)
