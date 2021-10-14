import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
import torch
import numpy as np
from torch.utils.data import DataLoader
from config import categories, categories_train, dataset, data_path, \
    device_ids, mix_model_path, dict_dir, layer, vMF_kappa, model_save_dir, \
        compnet_type, backbone_type, num_mixtures, da_mix_model_path, da_dict_dir
from config import config as cfg
from model import Net
from helpers import getImg, Imgset, imgLoader, getVmfKernels, getCompositionModel, update_clutter_model, UnNormalize
from model import resnet_feature_extractor
import tqdm
import torchvision.models as models

import torchvision
u = UnNormalize()

DA = True
test_orig = False  #* test compnets with clean images
corr = 'glass_blur'  # 'snow'

print("{} Corruption Model is {} and clean test data is {}".format(corr, DA, test_orig))

###################
# Test parameters #
###################
likely = 0.6  # occlusion likelihood
occ_levels = ['ZERO', 'ONE', 'FIVE', 'NINE'] # occlusion levels to be evaluated [0%,20-40%,40-60%,60-80%]
bool_load_pretrained_model = False # False if you want to load initialization (see Initialization_Code/)
bool_mixture_model_bg = False 	# use maximal mixture model or sum of all mixture models, not so important
bool_multi_stage_model = False 	# this is an old setup


def test(models, test_data, batch_size):
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    print('Testing')
    nclasses = models[0].num_classes
    # print(nclasses)
    correct = np.zeros(nclasses)
    total_samples = np.zeros(nclasses)
    scores = np.zeros((0,nclasses))

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            input, mask, label = data
            input.requires_grad = False

            # saving images
            # if i%500==0:
            #     im = torchvision.transforms.functional.to_pil_image(u(input[0]))
                # print(type(im), input.shape)
                # im.show()
            #     # im.save('results/1elas4_%s.png' % i)
            #     del im
                # exit()

            if device_ids:
                input = input.cuda(device_ids[0])
            c_label = label.numpy()

            # try:
            output, *_ = models[0](input)
            # except RuntimeError as e:
            #     print(e, input.shape)
            #     continue
            out = output.cpu().numpy()

            scores = np.concatenate((scores, out))
            out = out.argmax(1)
            correct[c_label] += np.sum(out == c_label)
            if out!= c_label and c_label == 5:
                im = torchvision.transforms.functional.to_pil_image(u(input[0]))
                im.save('results/fails/car_incorrect_%s_%s.png' % (corr, i))
                del im

            total_samples[c_label] += 1

    for i in range(nclasses):
        if total_samples[i]>0:
            print('Class {}: {:1.3f}'.format(categories_train[i],correct[i]/total_samples[i]))
    test_acc = (np.sum(correct)/np.sum(total_samples))
    return test_acc, scores

if __name__ == '__main__':

    if compnet_type=='bernoulli':
        bool_mixture_model_bg = True

    if bool_load_pretrained_model:
        print(">>>>>>>>>>>>> LOADING Pre-Trained Model")
        if backbone_type=='vgg':
            if layer=='pool5':
                if dataset=='pascal3d+':
                    vc_dir = model_save_dir+'vgg_pool5_p3d+/'
                elif dataset=='coco':
                    vc_dir = model_save_dir+'vgg_pool5_coco/'

        vc_file = vc_dir + 'best.pth'
        outdir = vc_dir
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
    else:
        print("Loading: ", dict_dir)
        print("Loading: ", mix_model_path)
        weights = getVmfKernels(dict_dir, device_ids)
        mix_models = getCompositionModel(device_ids, mix_model_path, layer, categories_train,\
            compnet_type=compnet_type,num_mixtures=num_mixtures)
    net = Net(extractor, weights, vMF_kappa, occ_likely, mix_models, bool_mixture_bg=bool_mixture_model_bg,\
        compnet_type=compnet_type, num_mixtures=num_mixtures, vc_thresholds=cfg.MODEL.VC_THRESHOLD)
    if device_ids:
        net = net.cuda(device_ids[0])
    nets=[]
    nets.append(net.eval())


    if bool_load_pretrained_model:
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

        for index, occ_type in enumerate(occ_types):
            # load images
            if test_orig == False:
                print("Loading corrupted data")
                test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
                    categories, occ_level, occ_type,bool_load_occ_mask=True, determinate=True, corruption=corr)  # masks is empty for some reason
            else:
                print("Loading Clean data")
                test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
                    categories, occ_level, occ_type,bool_load_occ_mask=True)  # masks is empty for some reason
            print('Total imgs for test of occ_level {} and occ_type {} '.format(occ_level, occ_type) + str(len(test_imgs)))
            # input()
            if DA is True and occ_level != 'ZERO':
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
            acc,scores = test(models=nets, test_data=test_imgset, batch_size=1)
            out_str = 'Model Name: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc)
            print(out_str)