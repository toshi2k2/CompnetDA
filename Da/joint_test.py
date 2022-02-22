#/ Using multiple VC models for prediction

import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
import torch
import numpy as np
from torch.utils.data import DataLoader
from Code.config import categories, categories_train, dataset, data_path, \
    device_ids, mix_model_path, dict_dir, layer, vMF_kappa, model_save_dir, \
        compnet_type, backbone_type, num_mixtures, da_mix_model_path, da_dict_dir
from Code.config import config as cfg
from Code.model import Net
from Code.helpers import getImg, Imgset, imgLoader, getVmfKernels, getCompositionModel, update_clutter_model, \
    UnNormalize
from Code.model import resnet_feature_extractor
import tqdm
import torchvision.models as models
from Initialization_Code.config_initialization import robin_cats
import torchvision, pickle

u = UnNormalize()

if dataset in ['robin','pseudorobin']:
    categories_train.remove('bottle')
    categories = categories_train
    cat = [robin_cats[0]]
    # cat = None
    print("Testing Sub-Category(ies) {}\n".format(cat))
else:
    cat = None

DA = True#False
test_orig = True#False  #* test compnets with clean images
corr = None#'snow'  # 'snow'
vc_space = 0#2,3 # default=0
vc_space2 = 0#3 # default=0
save_scores = False#True #!only implemented for one occlusion level
save_pseudo_image_list = False#True #!only implemented for one occlusion level

dataset= 'pseudorobin'#'pascal3d+'
backbone_type = 'vgg_bn' #'vgg_bn
da_dict_dir = 'models/da_init_{}/dictionary_vgg_bn/dictionary_pool5_512.pickle'.format(backbone_type)
da_mix_model_path = 'models/da_init_{}/mix_model_vmf_{}_EM_all'.format(backbone_type,dataset)
dict_dir = 'models_snow_bn/init_{}/dictionary_{}/dictionary_pool5_512.pickle'.format(backbone_type,backbone_type)
# dict_dir = 'models_old/init_{}/dictionary_{}/dictionary_pool5.pickle'.format(backbone_type,backbone_type)
mix_model_path = 'models_snow_bn/init_{}/mix_model_vmf_{}_EM_all'.format(backbone_type,dataset)
# da_dict_dir = 'models/da_init_vgg/dictionary_vgg/corres_dict_pool5_512.pickle'

dataset = 'robin'

# da_dict_dir2 = 'models_robin_all_3x3/da_init_vgg_bn/dictionary_vgg_bn/dictionary_pool5_512.pickle'
da_dict_dir2 = da_dict_dir
da_mix_model_path2 = 'models/da_init_{}/mix_model_vmf_{}_EM_all'.format(backbone_type,dataset)
dict_dir2 = 'models_snow_bn/init_{}/dictionary_{}/dictionary_pool5_512.pickle'.format(backbone_type,backbone_type)
mix_model_path2 = 'models_snow_bn/init_{}/mix_model_vmf_{}_EM_all'.format(backbone_type,dataset)

# dataset = 'robin'
print("{} Corruption Model is {} and clean test data is {}".format(corr, DA, test_orig))

###################
# Test parameters #
###################
likely = 0.6  # occlusion likelihood
occ_levels = ['ZERO', 'ONE', 'FIVE', 'NINE'] # occlusion levels to be evaluated [0%,20-40%,40-60%,60-80%]
if dataset == 'robin':
    occ_levels = ['ZERO']
bool_load_pretrained_model = False # False if you want to load initialization (see Initialization_Code/)
bool_mixture_model_bg = False 	# use maximal mixture model or sum of all mixture models, not so important
bool_multi_stage_model = False 	# this is an old setup


def test(models, models2, test_data, batch_size):
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    print('Testing')
    nclasses = models[0].num_classes
    # print(nclasses)
    correct = np.zeros(nclasses)
    total_samples = np.zeros(nclasses)
    scores = np.zeros((0,nclasses))
    scores2 = np.zeros((0,nclasses))
    real_labels = []
    pseudo_img_pth, pseudo_labels = [], []

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            input, mask, label = data
            input.requires_grad = False

            if device_ids:
                input = input.cuda(device_ids[0])
            c_label = label.numpy()

            real_labels.append(c_label)

            output, *_ = models[0](input)
            out = output.cpu().numpy()

            output2, *_ = models2[0](input)
            out2 = output2.cpu().numpy()

            scores = np.concatenate((scores, out))
            scores2 = np.concatenate((scores2, out2))

            st1 = np.argpartition(out[0], -2)[-2:]
            st2 = np.argpartition(out2[0], -2)[-2:] 

            if out.argmax(1)==out2.argmax(1):
                
                # if save_pseudo_image_list:
                #     mr = out.max(axis=1)
                #     res = np.sort(out, axis=1)[:, -2]
                #     ret = mr[0]-res[0] 
                #     if ret>0.05:
                #         pseudo_img_pth.append(test_data.images[i])
                #         pseudo_labels.append(out.argmax(1))

                out = out.argmax(1)
                # out2 = out2.argmax(1)
                correct[c_label] += np.sum(out == c_label)
                if out==c_label:
                    pseudo_img_pth.append(test_data.images[i])
                    pseudo_labels.append(out)
            # elif np.intersect1d(st1,st2).size!=0:
            #     if np.intersect1d(st1,st2).size == 2:
            #         if out.max(1)>out2.max(1):
            #             out = out.argmax(1)
            #         else:
            #             out = out2.argmax(1)
            #         # tm0, tm1 =np.intersect1d(st1,st2)
            #         # # print(out[tm0]+out2[tm0])
            #         # if out[0,tm0]+out2[0,tm0]>out[0,tm1]+out2[0,tm1]:
            #         #     out = tm0
            #         # else: out = tm1
            #     else:
            #         out = np.intersect1d(st1,st2)
            #         # out2 = out2.argmax(1)
            #     correct[c_label] += np.sum(out == c_label)
            #     if out==c_label:
            #         pseudo_img_pth.append(test_data.images[i])
            #         pseudo_labels.append(out)
            else:
                # if out.max(axis=1)>0.3:
                #     if out.argmax(1)!=5: #chair
                #         pseudo_img_pth.append(test_data.images[i])
                #         pseudo_labels.append(out.argmax(1))
                out = (out+out2)/2.
                out = out.argmax(1)
                correct[c_label] += np.sum(out == c_label)
                if out==c_label:
                    pseudo_img_pth.append(test_data.images[i])
                    pseudo_labels.append(out)
            # if out!= c_label and c_label == 5:
                # im = torchvision.transforms.functional.to_pil_image(u(input[0]))
                # im.save('results/fails/car_incorrect_%s_%s.png' % (corr, i))
                # im.save('test_1.png')
                # del im

            total_samples[c_label] += 1

    for i in range(nclasses):
        if total_samples[i]>0:
            print('Class {}: {:1.3f}'.format(categories_train[i],correct[i]/total_samples[i]))
    test_acc = (np.sum(correct)/np.sum(total_samples))
    if save_scores:
        np.savez('dual{}_da_{}_{}-{}.npz'.format(dataset, DA, vc_space, vc_space2), scores, scores2, np.array(real_labels))
    if save_pseudo_image_list:
        print("Total number of pseudo images = {} ({}%)".format(len(pseudo_img_pth), len(pseudo_img_pth)/sum(total_samples)))
        # np.savez('image_list_{}_da_{}.npz'.format(dataset, DA), np.array(pseudo_img_pth), np.array(pseudo_labels))
        with open("robin_all_psuedo_img2.pickle", 'wb') as fh:
            # print('saving at: '+savename)
            pickle.dump([pseudo_img_pth, pseudo_labels], fh)
    return test_acc, scores

if __name__ == '__main__':

    if compnet_type=='bernoulli':
        bool_mixture_model_bg = True

    if bool_load_pretrained_model:
        print(">>>>>>>>>>>>> LOADING Pre-Trained Model")
        if backbone_type=='vgg' or 'vgg_tr':
            if layer=='pool5':
                if dataset=='pascal3d+':
                    vc_dir = model_save_dir+'vgg_pool5_p3d+/'
                elif dataset=='coco':
                    vc_dir = model_save_dir+'vgg_pool5_coco/'

        vc_file = vc_dir + 'best.pth'
        outdir = vc_dir
        # vc_file = model_save_dir + 'train_pool5_a3_b3_vcTrue_mixTrue_occlikely0.6_vc512_lr_0.01_pascal3d+_pretrainedFalse_epochs_50_occFalse_backbonevgg_0/vc9.pth'
        vc_file = model_save_dir + 'train_pool5_a0_b0_vcTrue_mixTrue_occlikely0.6_vc512_lr_0.0001_pascal3d+_pretrainedFalse_epochs_50_occFalse_backbonevgg_tr_0/vc2.pth'
    else:
        if DA:
            vc_dir = model_save_dir+'da_compnet_{}_{}_{}_initialization/'.format(layer,\
                compnet_type,likely,dataset)
        else:
            vc_dir = model_save_dir+'compnet_{}_{}_{}_initialization/'.format(layer,\
                compnet_type,likely,dataset)
        print("VD Dir: ", vc_dir)
        vc_path = ''
        outdir = vc_dir

    # if not os.path.exists(outdir):
    #     print("creating :", outdir)
    #     os.makedirs(outdir)
    # info = outdir + 'config.txt'

    occ_likely = []
    for i in range(len(categories_train)):
        occ_likely.append(likely)

    ############################
    # Get CompositionalNet Init
    ############################
    if backbone_type=='vgg':
        if layer=='pool4':
            extractor = models.vgg16(pretrained=True).features[0:24]
        else:
            extractor = models.vgg16(pretrained=True).features
    elif backbone_type=='vgg_bn':
        if layer=='pool4':
            extractor = models.vgg16_bn(pretrained=True).features[0:24]
        else:
            extractor = models.vgg16_bn(pretrained=True).features
    elif backbone_type =='vgg_tr':
        saved_model = 'baseline_models/train_None_lr_0.01_pascal3d+_pretrained_False_epochs_15_occ_False_backbonevgg_0/vgg14.pth'
        # saved_model = 'baseline_models/snowadaptedvggbn.pth' # adapted vgg_bn
        saved_model = 'baseline_models/None_adapted_vgg_bn_robin.pth' # adapted vgg_bn for robin
        load_dict = torch.load(saved_model, map_location='cuda:{}'.format(0))
        # tmp = models.vgg16(pretrained=False)
        tmp = models.vgg16_bn(pretrained=False)
        num_ftrs = tmp.classifier[6].in_features
        tmp.classifier[6] = torch.nn.Linear(num_ftrs, len(categories_train))
        tmp.load_state_dict(load_dict['state_dict'])
        tmp.eval()
        if layer=='pool4':
            extractor = tmp.features[0:24]
        else:
            extractor = tmp.features
    elif backbone_type=='resnet50' or backbone_type == 'resnet18' or backbone_type == 'resnext' \
        or backbone_type=='densenet':
        extractor = resnet_feature_extractor(backbone_type, layer)

    extractor.cuda(device_ids[0]).eval()

    if DA:
        print("Loading: ", da_dict_dir)
        print("Loading: ", da_mix_model_path)
        weights = getVmfKernels(da_dict_dir, device_ids)
        mix_models = getCompositionModel(device_ids, da_mix_model_path, layer, categories_train, \
                                         compnet_type=compnet_type, num_mixtures=num_mixtures)
        print("Loading: ", da_dict_dir2)
        print("Loading: ", da_mix_model_path2)
        weights2 = getVmfKernels(da_dict_dir2, device_ids)
        mix_models2 = getCompositionModel(device_ids, da_mix_model_path2, layer, categories_train, \
                                         compnet_type=compnet_type, num_mixtures=num_mixtures)
    else:
        #! TO-Do
        raise(RuntimeError)
        print("Loading: ", dict_dir)
        print("Loading: ", mix_model_path)
        weights = getVmfKernels(dict_dir, device_ids)
        mix_models = getCompositionModel(device_ids, mix_model_path, layer, categories_train,\
            compnet_type=compnet_type,num_mixtures=num_mixtures)
    net = Net(extractor, weights, vMF_kappa, occ_likely, mix_models, bool_mixture_bg=bool_mixture_model_bg,\
        compnet_type=compnet_type, num_mixtures=num_mixtures, vc_thresholds=cfg.MODEL.VC_THRESHOLD, vc_sp=vc_space)
    net2 = Net(extractor, weights2, vMF_kappa, occ_likely, mix_models2, bool_mixture_bg=bool_mixture_model_bg,\
        compnet_type=compnet_type, num_mixtures=num_mixtures, vc_thresholds=cfg.MODEL.VC_THRESHOLD, vc_sp=vc_space2)
    if device_ids:
        net = net.cuda(device_ids[0])
        net2 = net2.cuda(device_ids[0])
    nets, nets2=[], []
    nets.append(net.eval())
    nets2.append(net2.eval())


    if bool_load_pretrained_model:
        raise(RuntimeError)
        if device_ids:
            load_dict = torch.load(vc_file, map_location='cuda:{}'.format(device_ids[0]))
        else:
            load_dict = torch.load(vc_file, map_location='cpu')
        net.load_state_dict(load_dict['state_dict'])
        if device_ids:
            net = net.cuda(device_ids[0])
        updated_clutter = update_clutter_model(net,device_ids)
        net.clutter_model = updated_clutter
        nets = []
        nets.append(net)

    ############################
    # Test Loop
    ############################
    for occ_level in occ_levels:

        if occ_level == 'ZERO':
            occ_types = ['']
        else:
            if dataset=='pascal3d+':
                occ_types = ['']#['_white','_noise', '_texture', '']
            elif dataset=='coco':
                occ_types = ['']
            elif dataset=='robin':
                occ_types = ['']

        for index, occ_type in enumerate(occ_types):
            # load images
            if test_orig == False:
                print("Loading corrupted test data")
                test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
                    categories, occ_level, occ_type,bool_load_occ_mask=True, determinate=True, corruption=corr)  # masks is empty for some reason
            else:
                print("Loading Clean test data")
                test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
                    categories, occ_level, occ_type,bool_load_occ_mask=True, subcat=cat)  # masks is empty for some reason
            print('Total imgs for test of occ_level {} and occ_type {} '.format(occ_level, occ_type) + str(len(test_imgs)))
            # input()
            if test_orig is False and occ_level != 'ZERO':
                errs = ["data/pascal3d+_occ_{}/carLEVEL{}/n03770679_14513_2.JPEG".format(corr, occ_level)]
                # errs = ['data/pascal3d+_occ_snow/carLEVELONE/n03770679_14513_2.JPEG', 'data/pascal3d+_occ_snow/carLEVELFIVE/n03770679_14513_2.JPEG', 'data/pascal3d+_occ_snow/carLEVELNINE/n03770679_14513_2.JPEG']
            else:
                errs = []
            for es in errs:
                if es in test_imgs:
                    idx_rm = test_imgs.index(es)
                    del test_imgs[idx_rm]
                    del test_labels[idx_rm]
                    del masks[idx_rm]
            """test_imgs is list of image path and name strings"""
            # get image loader
            test_imgset = Imgset(test_imgs, masks, test_labels, imgLoader, bool_square_images=False)
            # compute test accuracy
            acc,scores = test(models=nets, models2=nets2, test_data=test_imgset, batch_size=1)
            out_str = 'Model Name: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc)
            print(out_str)