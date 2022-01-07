"""Train with batchnorm adaptation and psuedo labelling"""
import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
from Code.model import Net
from Code.helpers import getImg, Imgset, imgLoader, save_checkpoint,getCompositionModel,getVmfKernels, \
    update_clutter_model
from Code.config import device_ids, mix_model_path, categories, categories_train, dict_dir, dataset, data_path, layer, \
    vc_num, model_save_dir, compnet_type,backbone_type, vMF_kappa,num_mixtures, da_dict_dir, da_mix_model_path
from Code.config import config as cfg
from torch.utils.data import DataLoader
from Code.losses import ClusterLoss
from Code.model import resnet_feature_extractor
import torchvision.models as models

import time
import torch
import torch.nn as nn
import numpy as np
import random
import robusta, tqdm
import copy

#---------------------
# Training Parameters
#---------------------
alpha = 0#3  # vc-loss
beta = 0#3 # mix loss
likely = 0.6 # occlusion likelihood
lr = 1e-2 # learning rate
batch_size = 1 # these are pseudo batches as the aspect ratio of images for CompNets is not square
# Training setup
vc_flag = False#True # train the vMF kernels
mix_flag = False#True # train mixture components
ncoord_it = 50 	#number of epochs to train

bool_mixture_model_bg = False #True: use a mixture of background models per pixel, False: use one bg model for whole image
bool_load_pretrained_model = False
bool_train_with_occluders = False

DA = True
corr = 'snow'
mode = 'mixed'
data_flag = 'corrupt' # clean, corrupt

gce = True  # general cross entropy loss 
bn_train = True  # batch_norm adaptation

# da_dict_dir = 'models/da_init_vgg_bn/dictionary_vgg_bn/corres_dict_pool5_512.pickle'
da_dict_dir = 'models_snow_bn/da_init_vgg_bn/dictionary_vgg_bn/dictionary_pool5_512.pickle'
da_mix_model_path = 'models_snow_bn/da_init_vgg_bn/mix_model_vmf_pascal3d+_EM_all'
backbone_type = 'vgg_bn'

occ_levels_test = ['ZERO', 'ONE', 'FIVE', 'NINE']
if bool_train_with_occluders:
    occ_levels_train = ['ZERO', 'ONE', 'FIVE', 'NINE']
else:
    occ_levels_train = ['ZERO']

out_dir = model_save_dir + 'train_{}_a{}_b{}_vc{}_mix{}_occlikely{}_vc{}_lr_{}_{}_pretrained{}_epochs_{}_occ{}_backbone{}_{}/'.format(
    layer, alpha,beta, vc_flag, mix_flag, likely, vc_num, lr, dataset, bool_load_pretrained_model,ncoord_it,bool_train_with_occluders,backbone_type,device_ids[0])


def test(models, test_data, batch_size):
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    print('Testing')
    nclasses = models.num_classes
    # print(nclasses)
    correct = np.zeros(nclasses)
    total_samples = np.zeros(nclasses)
    scores = np.zeros((0,nclasses))
    models.eval()

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            input, mask, label = data
            input.requires_grad = False

            if device_ids:
                input = input.cuda(device_ids[0])
            c_label = label.numpy()

            # try:
            output, *_ = models(input)
            out = output.cpu().numpy()

            scores = np.concatenate((scores, out))
            out = out.argmax(1)
            correct[c_label] += np.sum(out == c_label)
            # if out!= c_label and c_label == 5:
            #     im = torchvision.transforms.functional.to_pil_image(u(input[0]))
            #     im.save('results/fails/car_incorrect_%s_%s.png' % (corr, i))
            #     del im

            total_samples[c_label] += 1

    for i in range(nclasses):
        if total_samples[i]>0:
            print('Class {}: {:1.3f}'.format(categories_train[i],correct[i]/total_samples[i]))
    test_acc = (np.sum(correct)/np.sum(total_samples))
    return test_acc, scores


def test_loop(net, single_occ=False):
    ############################
    # Test Loop
    ############################
    if single_occ:
        occ_levels_test = ['ZERO']
    else:
        occ_levels_test = ['ZERO', 'ONE', 'FIVE', 'NINE']
    for occ_level in occ_levels_test:

        if occ_level == 'ZERO':
            occ_types = ['']
        else:
            if dataset=='pascal3d+':
                occ_types = ['']#['_white','_noise', '_texture', '']
            elif dataset=='coco':
                occ_types = ['']

        for index, occ_type in enumerate(occ_types):
            if data_flag == "corrupt":
                print("Loading corrupted test data")
                test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
                    categories, occ_level, occ_type,bool_load_occ_mask=True, determinate=True, corruption=corr)  # masks is empty for some reason
            else:
                print("Loading Clean test data")
                test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
                    categories, occ_level, occ_type,bool_load_occ_mask=True)  # masks is empty for some reason
            print('Total imgs for test of occ_level {} and occ_type {} '.format(occ_level, occ_type) + str(len(test_imgs)))
            # input()
            if data_flag =='corrupt' and occ_level != 'ZERO':
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
            acc,scores = test(models=net, test_data=test_imgset, batch_size=1)
            out_str = 'Model Name: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc)
            print(out_str)
    return


def train(model, train_data, val_data, epochs, batch_size, learning_rate, savedir, alpha=3,beta=3, \
    vc_flag=True, mix_flag=False, gce=False):
    best_check = {
        'epoch': 0,
        'best': 0,
        'val_acc': 0
    }
    out_file_name = savedir + 'result.txt'
    total_train = len(train_data)
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)
    val_loaders=[]

    for i in range(len(val_data)):
        val_loader = DataLoader(dataset=val_data[i], batch_size=1, shuffle=True)
        val_loaders.append(val_loader)

    # we observed that training the backbone does not make a very big difference but not training saves a lot of memory
    # if the backbone should be trained, then only with very small learning rate e.g. 1e-7
    # print(model.backbone)
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    
    if not vc_flag:
        model.conv1o1.weight.requires_grad = False
    else:
        model.conv1o1.weight.requires_grad = True

    if not mix_flag:
        model.mix_model.requires_grad = False
    else:
        model.mix_model.requires_grad = True

    if gce:
        print("Using GCE Loss\n")
        classification_loss = robusta.selflearning.GeneralizedCrossEntropy(q=0.8)
    else:   
        classification_loss = nn.CrossEntropyLoss()
    cluster_loss = ClusterLoss()

    for p in model.parameters():
        p.requires_grad = False

    if bn_train:
        for module in model.backbone.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                for name, parameter in module.named_parameters():
                    parameter.requires_grad = True

                    # print(module, name, parameter.requires_grad)
        # parameters = robusta.selflearning.adapt(model, adapt_type="affine")
        # print(list(parameters)[1])

    # print(model)
    # robusta.batchnorm.adapt(model, adapt_type="batch_wise")
    # parameters = robusta.selflearning.adapt(model, adapt_type="affine")
    # optimizer = torch.optim.SGD(parameters, lr=1e-3)
    optimizer = torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)
    # print(model)
    # exit()
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.98)
    o_model = copy.deepcopy(model)

    print('Training')
    # test_loop(model, True)
    model.train()
    for epoch in range(epochs):
        out_file = open(out_file_name, 'a')
        train_loss = 0.0
        correct = 0
        start = time.time()
        model.train()
        if not bn_train:
            model.backbone.eval()
        for index, data in enumerate(train_loader):
            if index % 500 == 0 and index != 0:
                end = time.time()
                print('Epoch{}: {}/{}, Acc: {}, Loss: {} Time:{}'.format(epoch + 1, index, total_train, correct.cpu().item() / index, train_loss.cpu().item() / index, (end-start)))
                # test_loop(model, True)
                start = time.time()

            input, _, label = data

            input = input.cuda(device_ids[0])
            label = label.cuda(device_ids[0])

            output, vgg_feat, like = model(input)

            # out = output.argmax(1)
            out = like.argmax(1)
            correct += torch.sum(out == label)
            if mode in [None, '']:
                class_loss = classification_loss(output, label) / output.shape[0]
            elif mode in ['mixed', 'corres']:
                # class_loss = classification_loss(output, out) / output.shape[0]
                class_loss = classification_loss(like, out) / output.shape[0]

            loss = class_loss
            if alpha != 0:
                clust_loss = cluster_loss(vgg_feat, model.conv1o1.weight) / output.shape[0]
                loss += alpha * clust_loss

            if beta!=0:
                if not gce:
                    mix_loss = like[0,label[0]]
                else:
                    mix_loss = like[0,out[0]]
                loss += -beta *mix_loss

            #with torch.autograd.set_detect_anomaly(True):
            loss.backward()

            # pseudo batches
            if np.mod(index,batch_size)==0:# and index!=0:
                optimizer.step()
                optimizer.zero_grad()

                models_differ = 0
                for key_item_1, key_item_2 in zip(model.state_dict().items(), o_model.state_dict().items()):
                    if torch.equal(key_item_1[1], key_item_2[1]):
                        pass
                    else:
                        models_differ += 1
                        if (key_item_1[0] == key_item_2[0]):
                            print('Mismatch found at', key_item_1[0])
                        else:
                            raise Exception
                exit()

                # for ((on,om),(n,m)) in zip(o_model.named_modules(),model.named_modules()):
                # for ((on,om),(n,m)) in zip(o_model.named_children(),model.named_children()):
                #     for ((n0,p0),(n1,p1)) in zip(om.named_parameters(),m.named_parameters()):
                #         assert(n0==n1)
                #         if not torch.equal(p0,p1) and not isinstance(om, torch.nn.BatchNorm2d):
                #             print(p0)
                #             print(p1)
                #             print(p0==p1)
                #             print(n0,type(om),":",on)
                #             print(n1,type(m),":",n)
                #             exit()
                # exit()

                

            train_loss += loss.detach() * input.shape[0]

        updated_clutter = update_clutter_model(model,device_ids)
        model.clutter_model = updated_clutter
        # scheduler.step()
        train_acc = correct.cpu().item() / total_train
        train_loss = train_loss.cpu().item() / total_train
        out_str = 'Epochs: [{}/{}], Train Acc:{}, Train Loss:{}'.format(epoch + 1, epochs, train_acc, train_loss)
        print(out_str)
        out_file.write(out_str)

        # Evaluate Validation images
        model.eval()
        with torch.no_grad():
            correct = 0
            val_accs=[]
            for i in range(len(val_loaders)):
                val_loader = val_loaders[i]
                correct_local=0
                total_local = 0
                val_loss = 0
                out_pred = torch.zeros(len(val_data[i].images))
                for index, data in enumerate(val_loader):
                    input,_, label = data
                    input = input.cuda(device_ids[0])
                    label = label.cuda(device_ids[0])
                    output,_,_ = model(input)
                    out = output.argmax(1)
                    out_pred[index] = out
                    correct_local += torch.sum(out == label)
                    total_local += label.shape[0]

                    class_loss = classification_loss(output, label) / output.shape[0]
                    loss = class_loss
                    val_loss += loss.detach() * input.shape[0]
                correct += correct_local
                val_acc = correct_local.cpu().item() / total_local
                val_loss = val_loss.cpu().item() / total_local
                val_accs.append(val_acc)
                out_str = 'Epochs: [{}/{}], Val-Set {}, Val Acc:{} Val Loss:{}\n'.format(epoch + 1, epochs,i , val_acc,val_loss)
                print(out_str)
                out_file.write(out_str)
            val_acc = np.mean(val_accs)
            out_file.write('Epochs: [{}/{}], Val Acc:{}\n'.format(epoch + 1, epochs, val_acc))
            if val_acc>best_check['val_acc']:
                print('BEST: {}'.format(val_acc))
                out_file.write('BEST: {}\n'.format(val_acc))
                best_check = {
                    'state_dict': model.state_dict(),
                    'val_acc': val_acc,
                    'epoch': epoch
                }
                save_checkpoint(best_check, savedir + 'vc' + str(epoch + 1) + '.pth', True)

            print('\n')
            test_loop(net, True)
        out_file.close()
    return best_check

if __name__ == '__main__':

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
    elif backbone_type=='vgg_tr':
        print("Loading vgg bn from a trained/adapted model - CHECK THE PATH\n")
        # saved_model = 'baseline_models/train_None_lr_0.01_pascal3d+_pretrained_False_epochs_15_occ_False_backbonevgg_0/vgg14.pth'
        saved_model = 'baseline_models/snowadaptedvggbn.pth' # adapted vgg_bn
        print("Loading pretrain model from {}".format(saved_model))

        load_dict = torch.load(saved_model, map_location='cuda:{}'.format(0))
        # tmp = models.vgg16(pretrained=False)
        tmp = models.vgg16_bn(pretrained=False)
        num_ftrs = tmp.classifier[6].in_features
        tmp.classifier[6] = torch.nn.Linear(num_ftrs, 12)
        tmp.load_state_dict(load_dict['state_dict'])
        # tmp.eval()
        if layer=='pool4':
            extractor = tmp.features[0:24]
        else:
            extractor = tmp.features
    elif backbone_type=='resnet50' or backbone_type=='resnext':
        extractor = resnet_feature_extractor(backbone_type, layer)
    else:
        raise RuntimeError("Unknown backbone")

    extractor.cuda(device_ids[0]).eval()
    if DA:
        print("Loading VC from ", da_dict_dir)
        weights = getVmfKernels(da_dict_dir, device_ids)
    else:
        weights = getVmfKernels(dict_dir, device_ids)

    if bool_load_pretrained_model:
        pretrained_file = 'PATH TO .PTH FILE HERE'
        # pretrained_file = 'models/train_pool5_a0_b0_vcTrue_mixTrue_occlikely0.6_vc512_lr_0.0001_pascal3d+_pretrainedFalse_epochs_50_occFalse_backbonevgg_tr_0/vc2.pth'
    else:
        pretrained_file = ''


    occ_likely = []
    for i in range(len(categories_train)):
        # setting the same occlusion likelihood for all classes
        occ_likely.append(likely)

    # load the CompNet initialized with ML and spectral clustering
    if DA:
        print("Loading Mixtures from: ", da_mix_model_path)
        mix_models = getCompositionModel(device_ids,da_mix_model_path,layer,categories_train,compnet_type=compnet_type)
    else:
        mix_models = getCompositionModel(device_ids,mix_model_path,layer,categories_train,compnet_type=compnet_type)
    net = Net(extractor, weights, vMF_kappa, occ_likely, mix_models, bool_mixture_bg=bool_mixture_model_bg,compnet_type=compnet_type,num_mixtures=num_mixtures, vc_thresholds=cfg.MODEL.VC_THRESHOLD)
    if bool_load_pretrained_model:
        print("Loading pretrained model")
        net.load_state_dict(torch.load(pretrained_file, map_location='cuda:{}'.format(device_ids[0]))['state_dict'])

    net = net.cuda(device_ids[0])

    train_imgs=[]
    train_masks = []
    train_labels = []
    val_imgs = []
    val_labels = []
    val_masks=[]

    # get training and validation images
    for occ_level in occ_levels_train:
        if occ_level == 'ZERO':
            occ_types = ['']
            train_fac=0.9
        else:
            occ_types = ['_white', '_noise', '_texture', '']
            train_fac=0.1

        for occ_type in occ_types:
            if data_flag == 'clean':
                imgs, labels, masks = getImg('train', categories_train, dataset, data_path, categories, occ_level, occ_type, bool_load_occ_mask=False)
            elif data_flag == 'corrupt':
                print("Loading corrupted data")
                imgs, labels, masks = getImg('train', categories_train, dataset, data_path, categories, occ_level, occ_type, \
                    bool_load_occ_mask=False, corruption=corr, determinate=True)
            else: raise(RuntimeError)

            nimgs=len(imgs)
            for i in range(nimgs):
                if (random.randint(0, nimgs - 1) / nimgs) <= train_fac:
                    train_imgs.append(imgs[i])
                    train_labels.append(labels[i])
                    train_masks.append(masks[i])
                elif not bool_train_with_occluders:
                    val_imgs.append(imgs[i])
                    val_labels.append(labels[i])
                    val_masks.append(masks[i])

    print('Total imgs for train ' + str(len(train_imgs)))
    print('Total imgs for val ' + str(len(val_imgs)))
    train_imgset = Imgset(train_imgs,train_masks, train_labels, imgLoader,bool_square_images=False)

    val_imgsets = []
    if val_imgs:
        val_imgset = Imgset(val_imgs,val_masks, val_labels, imgLoader,bool_square_images=False)
        val_imgsets.append(val_imgset)

    # write parameter settings into output folder
    load_flag = False
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    info = out_dir + 'config.txt'
    config_file = open(info, 'a')
    config_file.write(dataset)
    out_str = 'layer{}_a{}_b{}_vc{}_mix{}_occlikely{}_vc{}_lr{}/'.format(layer,alpha,beta,vc_flag,mix_flag,likely,vc_num,lr)
    config_file.write(out_str)
    out_str = 'Train\nDir: {}, vMF_kappa: {}, alpha: {},beta: {}, likely:{}\n'.format(out_dir, vMF_kappa, alpha,beta,likely)
    config_file.write(out_str)
    print(out_str)
    out_str = 'pretrain{}_file{}'.format(bool_load_pretrained_model,pretrained_file)
    print(out_str)
    config_file.write(out_str)
    config_file.close()

    # test_loop(net, True)

    train(model=net, train_data=train_imgset, val_data=val_imgsets, epochs=ncoord_it, batch_size=batch_size,
          learning_rate=lr, savedir=out_dir, alpha=alpha,beta=beta, vc_flag=vc_flag, mix_flag=mix_flag, gce=gce)


