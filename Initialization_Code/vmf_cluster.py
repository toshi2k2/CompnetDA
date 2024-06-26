#! Taking bigger spatial size for VCs

import os, sys

from cv2 import add
p = os.path.abspath('.')
sys.path.insert(1, p)
from Code.vMFMM import *
from config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, \
    Astride, Apad, Arf, vMF_kappa, layer, init_path, nn_type, dict_dir, offset, extractor, \
        da_dict_dir, robin_cats
from Code.helpers import UnNormalize, getImg, imgLoader, Imgset, myresize, UnNormalize
import torch
import torchvision
from torch.utils.data import DataLoader
import cv2
import glob
import pickle, random
import argparse

u = UnNormalize()

parser = argparse.ArgumentParser(description='vMF clustering')
parser.add_argument('--da', type=bool, default=False, help='running DA or not')
parser.add_argument('--corr', type=str, default=None, help='types of corruptions in dataset')
parser.add_argument('--cortest', type=bool, default=False, help='use corrupted test set as data - only for pascal3d+')
parser.add_argument('--back_cor', type=str, default=None, help='background corruption - if None, corruption is applied to entire image')
parser.add_argument('--robin_cat', type=int, default=None, help='None-all robin subcategories, else number')
parser.add_argument('--vcs', type=int, default=0, help='size of VCs used')
parser.add_argument('--retrain', type=bool, default=False, help='retrain VCs, then needs old VCs')
parser.add_argument('--addata', type=bool, default=False, help='add data from training data')
parser.add_argument('--frc', type=float, default=0.9, help='used when addata is True. Proportion of current data that is added')
parser.add_argument('--squareim', type=bool, default=False, help='use square images')
parser.add_argument('--dataset', type=str, default=None, help='None-dataset in config files-else the one you choose')

args = parser.parse_args()

if args.dataset is not None:
    dataset = args.dataset

if args.corr == None:
    assert(args.cortest==False)

DA = args.da #True
corr = args.corr# None#'snow'  # 'snow'
backgnd_corr = args.back_cor#None # backgnd corruption - if None corr will be applied to entire image
# cat  = [robin_cats[4]] #/ for individual sub-cats
if args.robin_cat is None:
    cat = args.robin_cat#None
else:
    cat=[robin_cats[args.robin_cat]]
# offset = 2 #/ should
vc_space = args.vcs#0#3
retrain_vc = args.retrain#True
add_data = args.addata#False
frc = args.frc#0.9
bool_square_images= args.squareim#True#False

img_per_cat = 1000  # image per category
if add_data:
    img_per_cat+=int(img_per_cat*frc)
samp_size_per_img = 20
imgs_par_cat =np.zeros(len(categories))
bool_load_existing_cluster = False
bins = 4

occ_level = 'ZERO'
occ_type = ''

print("Dataset {}".format(dataset))
if dataset == 'robin':
    categories.remove('bottle')
    cat_test = categories
    print("Category {}".format(cat))
else:
    cat_test = categories

#* imgs is the image paths array/list
if DA:
    if dataset=='robin':
        print("Loading {} test data".format(dataset))
        imgs, labels, masks = getImg('test', categories, dataset, data_path, cat_test, \
            occ_level, occ_type, bool_load_occ_mask=False, subcat=cat)    
    else:
        if args.cortest:
            print("Loading {} {} test data".format(dataset, corr))
            imgs, labels, masks = getImg('test', categories, dataset, data_path, cat_test, \
                occ_level, occ_type, bool_load_occ_mask=False, determinate=True, corruption=corr, \
                    corr_bck=backgnd_corr)
        else:
            print("Loading {} {} train data".format(dataset, corr))
            imgs, labels, masks = getImg('train', categories, dataset, data_path, cat_test, \
                occ_level, occ_type, bool_load_occ_mask=False, determinate=True, corruption=corr, \
                    corr_bck=backgnd_corr)
else:
    print("Loading {} train data".format(dataset))
    imgs, labels, masks = getImg('train', categories, dataset, data_path, cat_test, \
    occ_level, occ_type, bool_load_occ_mask=False)
#/############################################
if add_data:
    # assert(dataset=='robin')
    if dataset in ['robin']:
        print("Adding clean data from robin train!")
        imgs2, labels2, masks2 = getImg('train', categories, 'robin', data_path, cat_test, \
            occ_level, occ_type, bool_load_occ_mask=False)
    elif dataset in ['pascal3d+']:
        imgs2, labels2, masks2 = getImg('train', categories, dataset, data_path, cat_test, \
                occ_level, occ_type, bool_load_occ_mask=False, determinate=True, corruption=corr, \
                    corr_bck=backgnd_corr)
    # temp_list = list(zip(imgs2, labels2, masks2))
    # sampled = random.sample(temp_list, int(min(len(imgs), len(imgs2))*frc))
    # imgs+=[i for (i,l,m) in sampled]
    # labels+=[l for (i,l,m) in sampled]
    # masks+=[m for (i,l,m) in sampled] 
    imgs+=imgs2
    labels+=labels2
    masks+=masks2
#/#########################################
imgset = Imgset(imgs, masks, labels, imgLoader, bool_square_images=bool_square_images)
data_loader = DataLoader(dataset=imgset, batch_size=1, shuffle=False)
nimgs = len(imgs)

loc_set = []
feat_set = []
nfeats = 0
for ii, data in enumerate(data_loader):
    input, mask, label = data
    if np.mod(ii, 500) == 0:
        print('{} / {}'.format(ii, len(imgs)))
        im = torchvision.transforms.functional.to_pil_image(u(input[0]))
        im.save('results/vmf_cluster.png')
        del im

    fname = imgs[ii]
    category = labels[ii]

    if imgs_par_cat[label] < img_per_cat:
        with torch.no_grad():
            tmp = extractor(input.cuda(device_ids[0]))[0].detach().cpu().numpy()  # ends with Conv2d, Relu and MaxPool2D - [512, 7, 22]
        height, width = tmp.shape[1:3]  # 7, 22
        img = cv2.imread(imgs[ii])

        if vc_space in [2,3]:
            tmp2 = np.array(tmp)

        s1, s2 = height - 2*offset, width - 2*offset
        l, r = [], []
        if s1<1:
            if s2<1:
                continue
                tmp = tmp[:,offset-1:offset, offset:offset+1]
                l, r = [offset-1,offset], [offset,offset+1]
            else:
                continue
                tmp = tmp[:,offset-1:offset, offset:width - offset]
                l, r = [offset-1,offset], [offset,width - offset]
        elif s2<1:
            continue
            tmp = tmp[:,offset:height - offset, offset:offset+1]
            l, r = [offset,height - offset], [offset,offset+1]
        else:
            tmp = tmp[:,offset:height - offset, offset:width - offset]  # * taking a central crop [512, 1, 16]
            l, r = [offset,height - offset], [offset,width - offset]
        gtmp = tmp.reshape(tmp.shape[0], -1) # [512, 16] - 2D to 1D per channel
        # continue
        if gtmp.shape[1] >= samp_size_per_img:
            rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img]
        else:
            rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img - gtmp.shape[1]]
            # rand_idx = np.append(range(gtmp.shape[1]), rand_idx)
        tmp_feats = gtmp[:, rand_idx].T  # [4, 512] - randomly chosen 4 spatial points?
        patches = []
        #/ Calculating bigger spatial VC feature patches
        if vc_space == 3:
            tmp2 = tmp2[:, l[0]-1:l[1]+1, r[0]-1:r[1]+1] #/ Taking +2 colums and rows around tmp for a 3x3 window
            for rr in rand_idx:
                if l[1]-l[0]==1:
                    feat_patch = tmp2[:, :, rr:3+rr]
                elif r[1]-r[0]==1:
                    feat_patch = tmp2[:, rr:3+rr, :]
                elif r[1]-r[0]>2 and l[1]-l[0]>2:
                    ry = int(rr % s2) 
                    rx = int(rr / s2)
                    feat_patch = tmp2[:, rx:3+rx, ry:3+ry]
                else:
                    raise(RuntimeError)
                feat_patch = feat_patch.reshape(feat_patch.shape[0]*9, -1)
                patches.append(feat_patch)
            patches = np.asarray(patches)
            tmp_feats2 = patches.reshape(patches.shape[0],-1)
        if vc_space == 2:
            # if s1==1 and s2==1:
            #     print(s1,s2)
            if r[1]-r[0]==1:
                tmp2 = tmp2[:,offset-1:(height+1) - offset, offset:width - offset]
            elif l[1]-l[0]==1:
                tmp2 = tmp2[:,offset:height - offset, offset-1:(width+1) - offset]
            gtmp2 = tmp2.reshape(tmp2.shape[0], -1) # [512, 16] - 2D to 1D per channel
            tmp_feats2 = gtmp2[:, rand_idx+1]
            tmp_feats2pre = gtmp2[:, rand_idx]
            tmp_feats2post = gtmp2[:, rand_idx+2]
            # print((tmp_feats-tmp_feats2.T).sum())
            # print((tmp_feats-tmp_feats2pre.T).sum())
            assert(np.array_equal(tmp_feats, tmp_feats2.T))
            tmp_feats2 = np.concatenate((tmp_feats2, tmp_feats2pre, tmp_feats2post)).T

        # if tmp.shape[1]!=3 or tmp.shape[2]<3:
        # if tmp.shape[1]<1 or tmp.shape[2]<1:
        #     print(tmp.shape, height, width, gtmp.shape, s1)
        #     print(rand_idx)
        #     print(tmp_feats)
        # continue
        cnt = 0
        for rr in rand_idx:
            if s1==0:
                ihi, iwi = np.unravel_index(rr, (1, width - 2 * offset))
                if s2==0:
                    ihi, iwi = np.unravel_index(rr, (1, 1))
            elif s2==0:
                ihi, iwi = np.unravel_index(rr, (height - 2 * offset, 1))
            else:
                ihi, iwi = np.unravel_index(rr, (height - 2 * offset, width - 2 * offset)) # unravel 1D to 2D - rr is the 1D location
            hi = (ihi+offset)*(input.shape[2]/height)-Apad  # * seems like x,y position calculation - why?
            wi = (iwi + offset)*(input.shape[3]/width)-Apad
            # hi = Astride * (ihi + offset) - Apad
            # wi = Astride * (iwi + offset) - Apad

            # assert (hi >= 0)
            # assert (wi >= 0)
            # assert (hi <= img.shape[0] - Arf)
            # assert (wi <= img.shape[1] - Arf)
            loc_set.append([category, ii, hi, wi, hi+Arf, wi+Arf])  # * Arf is receptive field size - this is defining a rectangle patch?
            if vc_space in [2,3]:
                feat_set.append(tmp_feats2[cnt, :])    
            else:
                feat_set.append(tmp_feats[cnt, :])  # * element is size 512, - appends single spatial point feature for a single image from VGG16
            cnt += 1

        imgs_par_cat[label] += 1
# exit()

feat_set = np.asarray(feat_set)  # * [48050, 512]
loc_set = np.asarray(loc_set).T  # * [6, 48050]

print(feat_set.shape)
model = vMFMM(vc_num, 'k++')  # * k-means clustering - finding means and saving them in a pickle file
if not retrain_vc:
    model.fit(feat_set, vMF_kappa, max_it=150)
else:
    print("RETRAINING VCs\n")
    with open('models/FINAL/init_vgg_tr/dictionary_vgg_tr/dictionary_pool5_512.pickle', 'rb') as fh:
        prev_mu = pickle.load(fh)
    with open('models/FINAL/init_vgg_tr/dictionary_vgg_tr/dictionary_pool5_512_p.pickle', 'rb') as fh:
        prev_p = pickle.load(fh)
    prev_pi = np.sum(prev_p, axis=0)/prev_p.shape[0]
    # model.fit_soft(features=feat_set, p=prev_p, mu=prev_mu, pi=prev_pi, kappa=vMF_kappa, max_it=150)
    model.fit_map(feat_set, pre_mu=prev_mu, pre_pi=prev_pi, reg=0.,kappa=vMF_kappa, max_it=300)
    del(prev_p, prev_mu, prev_pi)

if DA:
    with open(da_dict_dir+'dictionary_{}_{}.pickle'.format(layer, vc_num), 'wb') as fh:
        pickle.dump(model.mu, fh)
else:
    with open(dict_dir+'dictionary_{}_{}.pickle'.format(layer, vc_num), 'wb') as fh:
        pickle.dump(model.mu, fh)


num = 50
SORTED_IDX = []
SORTED_LOC = []
for vc_i in range(vc_num):
    sort_idx = np.argsort(-model.p[:, vc_i])[0:num]
    SORTED_IDX.append(sort_idx)
    tmp=[]
    for idx in range(num):
        iloc = loc_set[:, sort_idx[idx]]
        tmp.append(iloc)
    SORTED_LOC.append(tmp)

if DA:
    with open(da_dict_dir + 'dictionary_{}_{}_p.pickle'.format(layer, vc_num), 'wb') as fh:
        pickle.dump(model.p, fh)
else:
    with open(dict_dir + 'dictionary_{}_{}_p.pickle'.format(layer, vc_num), 'wb') as fh:
        pickle.dump(model.p, fh)
p = model.p

print('save top {0} images for each cluster'.format(num))
example = [None for vc_i in range(vc_num)]
if DA:
    out_dir = da_dict_dir + '/cluster_images_{}_{}/'.format(layer, vc_num)
else:
    out_dir = dict_dir + '/cluster_images_{}_{}/'.format(layer, vc_num)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

print('')

for vc_i in range(vc_num):
    patch_set = np.zeros(((Arf**2)*3, num)).astype('uint8')
    sort_idx = SORTED_IDX[vc_i]  # np.argsort(-p[:,vc_i])[0:num]
    opath = out_dir + str(vc_i) + '/'
    if not os.path.exists(opath):
        os.makedirs(opath)
    locs=[]
    for idx in range(num):
        iloc = loc_set[:,sort_idx[idx]]
        category = iloc[0]
        loc = iloc[1:6].astype(int)
        if not loc[0] in locs:
            locs.append(loc[0])
            img = cv2.imread(imgs[int(loc[0])])
            img = myresize(img, 224, 'short')
            patch = img[loc[1]:loc[3], loc[2]:loc[4], :]
            # patch_set[:,idx] = patch.flatten()
            if patch.size:
                cv2.imwrite(opath+str(idx)+'.JPEG',patch)
    # example[vc_i] = np.copy(patch_set)
    if vc_i % 10 == 0:
        print(vc_i)

# print summary for each vc
# if layer=='pool4' or layer =='last': # somehow the patches seem too big for p5
for c in range(vc_num):
    iidir = out_dir + str(c) + '/'
    files = glob.glob(iidir+'*.JPEG')
    width = 100
    height = 100
    canvas = np.zeros((0, 4 * width, 3))
    cnt = 0
    for jj in range(4):
        row = np.zeros((height, 0, 3))
        ii = 0
        tries = 0
        next = False
        for ii in range(4):
            if (jj*4+ii) < len(files):
                img_file = files[jj*4+ii]
                if os.path.exists(img_file):
                    img = cv2.imread(img_file)
                img = cv2.resize(img, (width, height))
            else:
                img = np.zeros((height, width, 3))
            row = np.concatenate((row, img), axis=1)
        canvas = np.concatenate((canvas, row), axis=0)

    cv2.imwrite(out_dir+str(c)+'.JPEG', canvas)
