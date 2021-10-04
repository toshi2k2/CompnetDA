import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from Code.vMFMM import *
from config_initialization import vc_num, dataset, categories, data_path, cat_test, device_ids, \
    Astride, Apad, Arf, vMF_kappa, layer, init_path, nn_type, dict_dir, offset, extractor, \
        da_dict_dir
from Code.helpers import UnNormalize, getImg, imgLoader, Imgset, myresize, UnNormalize
import torch
import torchvision
from torch.utils.data import DataLoader
import cv2
import glob
import pickle

u = UnNormalize()

DA = False
corr = 'pixelate'  # 'snow'

img_per_cat = 1000  # image per category
samp_size_per_img = 20
imgs_par_cat =np.zeros(len(categories))
bool_load_existing_cluster = False
bins = 4

occ_level = 'ZERO'
occ_type = ''

#* imgs is the image paths array/list
if DA:
    imgs, labels, masks = getImg('train', categories, dataset, data_path, cat_test, \
        occ_level, occ_type, bool_load_occ_mask=False, determinate=True, corruption=corr)
else:
    imgs, labels, masks = getImg('train', categories, dataset, data_path, cat_test, \
    occ_level, occ_type, bool_load_occ_mask=False)
imgset = Imgset(imgs, masks, labels, imgLoader, bool_square_images=False)
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

        tmp = tmp[:,offset:height - offset, offset:width - offset]  # * taking a central crop [512, 1, 16]
        gtmp = tmp.reshape(tmp.shape[0], -1) # [512, 16] - 2D to 1D per channel
        if gtmp.shape[1] >= samp_size_per_img:
            rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img]
        else:
            rand_idx = np.random.permutation(gtmp.shape[1])[:samp_size_per_img - gtmp.shape[1]]
            # rand_idx = np.append(range(gtmp.shape[1]), rand_idx)
        tmp_feats = gtmp[:, rand_idx].T  # [4, 512] - randomly chosen 4 spatial points?

        cnt = 0
        for rr in rand_idx:
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
            feat_set.append(tmp_feats[cnt, :])  # * element is size 512, - appends single spatial point feature for a single image from VGG16
            cnt += 1

        imgs_par_cat[label] += 1


feat_set = np.asarray(feat_set)  # * [48050, 512]
loc_set = np.asarray(loc_set).T  # * [6, 48050]

print(feat_set.shape)
model = vMFMM(vc_num, 'k++')  # * k-means clustering - finding means and saving them in a pickle file
model.fit(feat_set, vMF_kappa, max_it=150)
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
