import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist
import pickle
import numpy as np
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, \
    cat_test, device_ids, Astride, Apad, Arf,vMF_kappa, layer,init_path, \
        dict_dir, sim_dir, da_sim_dir, extractor, model_save_dir, \
            da_dict_dir, da_init_path, robin_cats
from Code.helpers import getImg, imgLoader, Imgset
from torch.utils.data import DataLoader
import cv2
import gc
import matplotlib.pyplot as plt
import scipy.io as sio

DA = True
mode = '' # '', mixed, None, corres
corr = None #'snow'  #'snow'
vc_space = 0#3
cat=None

if dataset in ['robin','pseudorobin']:
    categories.remove('bottle')
    cat_test = categories
    # cat = [robin_cats[0]]
    cat = None

if DA and mode in ['corres']:
    print("Loading CORRESPONDENCE Vmf Kernels")
    dictfile=da_dict_dir+'corres_dict_{}_{}.pickle'.format(layer, vc_num)
elif DA or mode == 'mixed':
    print("Loading DA Vmf Kernels")
    dictfile=da_dict_dir+'dictionary_{}_{}.pickle'.format(layer,vc_num)
else:
    dictfile=dict_dir+'dictionary_{}_{}.pickle'.format(layer,vc_num)
print('loading {}'.format(dictfile))
with open(dictfile, 'rb') as fh:
    centers = pickle.load(fh)
#####################
# BEWARE
#####################

if DA or mode == 'mixed':
    mixdir = da_init_path + 'mix_model_vmf_{}_EM_all/'.format(dataset)
else:
    mixdir = init_path + 'mix_model_vmf_{}_EM_all/'.format(dataset)

occ_level='ZERO'
occ_type=''

print("Loading {} and {}\n".format(dictfile, mixdir))

mixs = {}
for ix, category in enumerate(categories):
    d1_savename = os.path.join(mixdir,'mmodel_{}_K4_FEATDIM{}_{}_specific_view.pickle'.format(category, vc_num, layer))
    print("Loading previous Mixtures\n {}".format(d1_savename))
    with open(d1_savename, 'rb') as fh:
        mixs[category]=pickle.load(fh)

mixers = {}
for ih,category in enumerate(categories):
    buk = []
    alpha = mixs[category]
    for i in range(len(alpha)):
        S = alpha[i]
        _,h,w = S.shape
        mini = []
        for ix in range(S.shape[0]):
            t = S[ix,:,:]
            # if np.max(t)>0.95:
            fn = np.sum(t>0.95)
            if fn>0:
                # mini.append((ix, np.round_((fn/(h*w))*100, decimals=3)))
                mini.append(ix)
                # print(ix)
        # mini.sort(key=lambda x: x[1])
        buk.append(mini)
    mixers[category]=buk

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def vc_set(data, cutoff=0.95):
    mini=[]
    for ix in range(data.shape[0]):
        t = data[ix,:,:]
        fn = np.sum(t>cutoff)
        if fn>0:
            mini.append(ix)
    return mini

def pad_shape(data, max_1, max_2, max_0=vc_num):
    vnum, ww, hh = data.shape
    assert (vnum == max_0)
    diff_w1 = int((max_1 - ww) / 2)
    diff_w2 = int(max_1 - ww - diff_w1)
    assert (max_1 == diff_w1 + diff_w2 + ww)
    diff_h1 = int((max_2 - hh) / 2)
    diff_h2 = int(max_2 - hh - diff_h1)
    assert (max_2 == diff_h1 + diff_h2 + hh)
    padded = np.pad(data, ((0, 0), (diff_w1, diff_w2), (diff_h1, diff_h2)), 'constant',constant_values=0)
    return padded

def learn_mix_model_vMF(category):
    
    if DA and mode not in ['mixed', 'corres']:
        if dataset in ['robin','pseudorobin']:
            print("Loading Robin test data (pseudo {})".format(dataset=='pseudorobin'))
            imgs, labels, masks = getImg('test', cat_test, dataset, data_path, [category], \
                occ_level, occ_type, bool_load_occ_mask=False, subcat=cat) 
        else:
            raise RuntimeError
            print("Loading corrupted data")
            imgs, labels, masks = getImg('train', [category], dataset, data_path, cat_test, \
                occ_level, occ_type, bool_load_occ_mask=False, determinate=True, corruption=corr)
    else:
        raise RuntimeError
        print("Loading clean data")
        imgs, labels, masks = getImg('train', [category], dataset, data_path, cat_test, \
            occ_level, occ_type, bool_load_occ_mask=False)

    print('total number of instances for obj {}: {}'.format(category, len(imgs)))
    N = len(imgs)
    N=10
    img_idx = np.asarray([nn for nn in range(N)])

    imgset = Imgset(imgs, masks, labels, imgLoader, bool_square_images=False)
    data_loader = DataLoader(dataset=imgset, batch_size=1, shuffle=False)

    r_set = []#[None for nn in range(N)]
    #layer_features 	  =	np.zeros((N,featDim,max_1,max_2),dtype=np.float32)
    for ii,data in enumerate(data_loader):
        if np.mod(ii,100)==0:
            print('{} / {}'.format(ii,N))
        input, mask, label = data
        layer_feature = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()
        iheight,iwidth = layer_feature.shape[1:3]

        if vc_space in [3,2]:
            if vc_space==3: ds = 9
            elif vc_space==2: ds = 3
            else: raise(RuntimeError)
            # layer_feature = np.pad(layer_feature)
            new_arr = np.zeros((layer_feature.shape[0]*ds, layer_feature.shape[1], layer_feature.shape[2]))
            x_ = np.pad(layer_feature, ((0, 0), (1, 1), (1, 1)), mode='edge') 
            for i in np.arange(1,x_.shape[1]-1):
                for j in np.arange(1,x_.shape[2]-1):
                    if vc_space == 3:
                        t = x_[:,i-1:i+2,j-1:j+2]
                    elif vc_space == 2:
                        t = x_[:,i-1:i+2,:]
                    t = t.reshape(t.shape[0]*ds, -1)
                    new_arr[:,i-1,j-1]=t[:,0]
            lff = new_arr.reshape(new_arr.shape[0], -1).T
            del new_arr, x_
        else:
            lff = layer_feature.reshape(layer_feature.shape[0],-1).T
        lff_norm = lff / (np.sqrt(np.sum(lff ** 2, 1)+1e-10).reshape(-1, 1))+1e-10
        # compute dot product
        tmp = (1-cdist(lff_norm, centers, 'cosine').astype(np.float32))
        # compute vMF likelihood
        tmp = tmp
        tmp = np.exp(vMF_kappa*tmp)
        # reshape such that the spatial position is preserved during learning
        feat_map = tmp.reshape(iheight, iwidth, -1).astype(np.float32).T
        r_set.append(feat_map)
        if ii==N:
            break

    # # num cluster centers
    # max_0 = vc_num
    # # width
    # max_1 = max([r_set[nn].shape[1] for nn in range(N)])
    # max_1=48
    # # height
    # max_2 = max([r_set[nn].shape[2] for nn in range(N)])
    # max_2=12
    # print(max_0, max_1, max_2)
    # layer_feature_vmf = np.zeros((N, max_0, max_1, max_2), dtype=np.float32)

    # for nn in range(N):
    #     vnum, ww, hh = r_set[nn].shape
    #     assert (vnum == max_0)
    #     diff_w1 = int((max_1 - ww) / 2)
    #     diff_w2 = int(max_1 - ww - diff_w1)
    #     assert (max_1 == diff_w1 + diff_w2 + ww)
    #     diff_h1 = int((max_2 - hh) / 2)
    #     diff_h2 = int(max_2 - hh - diff_h1)
    #     assert (max_2 == diff_h1 + diff_h2 + hh)
    #     padded = np.pad(r_set[nn], ((0, 0), (diff_w1, diff_w2), (diff_h1, diff_h2)), 'constant',constant_values=0)
    #     r_set[nn] = []
    #     layer_feature_vmf[nn,:,:,:] = padded


    '''
    # ML updates of mixture model and vMF mixture coefficients
    '''
    #compute feature likelihood
    for nn in range(N):
        # if nn % 100 == 0:
        print('############\nImage {}\n##########'.format(nn))
        for caty in categories:
            print("\nCategory-{}\n".format(caty))
            alphas = mixs[caty]
            for kk in range(len(alphas)):
                alpha=alphas[kk]
                # print(len(vc_set(alpha)))
                alpha[alpha<=.9]=0.
                # print(len(vc_set(alpha)))
                try:
                    vmf_feat = pad_shape(data=r_set[nn], max_1=alpha.shape[1], max_2=alpha.shape[2])
                except ValueError as e:
                    print("Can't pad.")
                    continue
                # like_map = layer_feature_vmf[img_idx[nn]]*alpha
                like_map = vmf_feat*alpha
                if np.sum(like_map)==0:
                    continue
                norm_like_map = NormalizeData(like_map)
                vcs = vc_set(norm_like_map, cutoff=0.5)
                # vcs = vc_set(NormalizeData(r_set[nn]), cutoff=0.1)
                assert(len(vcs)!=0)
                gd_alpha = mixers[caty][kk]
                sset = set(vcs).issubset(set(gd_alpha))
                outset = set(vcs).difference(set(gd_alpha))
                inset = set(vcs).intersection(set(gd_alpha))
                gdn = len(set(gd_alpha))
                if gdn == 0:
                    continue
                assert(gdn!=0)
                print("Mixture-{}: {} TrueSubset-{}, OutOfSet-{}({}%), MatchingSet-{}({}%)".format(kk, len(vcs), sset, len(outset), \
                    (len(outset)/len(vcs))*100., len(inset), (len(inset)/len(vcs))*100.))
                # likeli = np.sum(like_map, axis=0)+1e-10
                # mixture_likeli[nn, kk] = np.sum(np.log(likeli))

    #compute new mixture assigment for feature map

if __name__=='__main__':
    for category in categories:
        learn_mix_model_vMF(category)

    print('DONE')