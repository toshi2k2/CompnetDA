from cProfile import label
import os, sys
# from Code.test import DA
p = os.path.abspath('.')
sys.path.insert(1, p)
from model import Net
from helpers import getImg, Imgset, imgLoader, save_checkpoint,getCompositionModel,getVmfKernels, \
    update_clutter_model
from config import device_ids, mix_model_path, categories, categories_train, dict_dir, dataset, data_path, layer, \
    vc_num, model_save_dir, compnet_type,backbone_type, vMF_kappa,num_mixtures,\
        da_dict_dir, da_mix_model_path
from config import config as cfg
from torch.utils.data import DataLoader
from losses import ClusterLoss
from model import resnet_feature_extractor
import torchvision.models as models

import time
import torch
import torch.nn as nn
import numpy as np
import random
import robusta
import argparse

parser = argparse.ArgumentParser(description='Mixture Model Calculation')
parser.add_argument('--da', type=bool, default=False, help='running DA or not')
parser.add_argument('--corr', type=str, default=None, help='types of corruptions in dataset')
parser.add_argument('--mode', type=str, default='', help="DA mode - None, mixed,'',reverse")
parser.add_argument('--robin_cat', type=int, default=None, help='None-all robin subcategories, else number')
parser.add_argument('--epoch', type=int, default=60, help='number of epochs')
parser.add_argument('--tasval', type=bool, default=True, help='use test data for checks')
parser.add_argument('--gce', type=bool, default=True, help='use GCE loss')
# parser.add_argument('--frc', type=float, default=0.9, help='used when addata is True. Proportion of current data that is added')
parser.add_argument('--squareim', type=bool, default=True, help='use square images')
parser.add_argument('--dataset', type=str, default=None, help='None-dataset in config files-else the one you choose')
parser.add_argument('--bn_adapt', type=bool, default=False, help='Use Batchnorm adapt')
parser.add_argument('--ctrain', type=bool, default=False, help='Continue training')
parser.add_argument('--ctrainpth', type=str, default=None, help='Continue training model path')
parser.add_argument('--tr_model', type=str, default=None, help='Load pretrained backbone-only') #!Caution

args = parser.parse_args()

if args.dataset is not None:
    dataset = args.dataset

robin_cats = ['context', 'weather', 'texture', 'pose', 'shape']

if dataset in ['robin', 'pseudorobin']:
    categories_train.remove('bottle')
    categories = categories_train
    # cat = [robin_cats[2]]
    if args.robin_cat is None:
        cat = args.robin_cat#None
    else:
        cat=[robin_cats[args.robin_cat]]
    # cat = None
    print("Testing Sub-Category(ies) {}\n".format(cat))
else:
    cat = None
    categories = categories_train
    corr = args.corr
#---------------------
# Training Parameters
#---------------------
alpha = 3  # vc-loss
beta = 3 # mix loss
likely = 0.6 # occlusion likelihood
lr = 1e-2 # learning rate

batch_size = 64#1 # these are pseudo batches as the aspect ratio of images for CompNets is not square
pseudo_batch_size=64
DA=args.da#True
mode=args.mode#'mixed' # ''-train with test set, 'mixed' - training data
test_as_val=args.tasval#True # use test data for validation
use_gce = args.gce#False#True

# Training setup
vc_flag = True # train the vMF kernels
mix_flag = True # train mixture components
ncoord_it = args.epoch#60 	#number of epochs to train

bool_mixture_model_bg = False #True: use a mixture of background models per pixel, False: use one bg model for whole image
bool_load_pretrained_model = args.ctrain#False
bool_train_with_occluders = False
bool_square_images = args.squareim#True#False

dataset= 'pseudopascal'#'robin'#'pascal3d+'pseudorobin','pseudopascal' #/ needed for robin
# backbone_type = 'vgg_tr' #'vgg_bn, 'vgg_tr'
# layer='last'
da_dict_dir = 'models/da_init_{}/dictionary_{}/dictionary_{}_{}.pickle'.format(backbone_type, backbone_type, layer, vc_num)
da_mix_model_path = 'models/da_init_{}/mix_model_vmf_{}_EM_all'.format(backbone_type,dataset)
dict_dir = 'models/init_{}/dictionary_{}/dictionary_{}_{}.pickle'.format(backbone_type,backbone_type, layer, vc_num)
# dict_dir = 'models_old/init_{}/dictionary_{}/dictionary_pool5.pickle'.format(backbone_type,backbone_type)
mix_model_path = 'models/init_{}/mix_model_vmf_{}_EM_all'.format(backbone_type,dataset)
# da_dict_dir = 'models/da_init_vgg/dictionary_vgg/corres_dict_pool5_512.pickle'

dataset='pascal3d+'#'robin' #/ needed for robin

if bool_train_with_occluders:
    occ_levels_train = ['ZERO', 'ONE', 'FIVE', 'NINE']
else:
    occ_levels_train = ['ZERO']

# out_dir = model_save_dir + 'train_{}_a{}_b{}_vc{}_mix{}_occlikely{}_vc{}_lr_{}_{}_pretrained{}_epochs_{}_occ{}_backbone{}_{}/'.format(
# 	layer, alpha,beta, vc_flag, mix_flag, likely, vc_num, lr, dataset, bool_load_pretrained_model,ncoord_it,bool_train_with_occluders,backbone_type,device_ids[0])
if cat!=None:
    out_dir = model_save_dir + '/vc{}{}_final/'.format(backbone_type, cat[0])
else: out_dir = model_save_dir + '/vc{}_final/'.format(backbone_type)


def train(model, train_data, val_data, epochs, batch_size, learning_rate, savedir, alpha=3,beta=3, vc_flag=True, mix_flag=False):
    best_check = {
        'epoch': 0,
        'best': 0,
        'val_acc': 0
    }
    out_file_name = savedir + 'result.txt'
    total_train = len(train_data)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loaders=[]

    for i in range(len(val_data)):
        val_loader = DataLoader(dataset=val_data[i], batch_size=1, shuffle=False)
        val_loaders.append(val_loader)

    # we observed that training the backbone does not make a very big difference but not training saves a lot of memory
    # if the backbone should be trained, then only with very small learning rate e.g. 1e-7
    if args.bn_adapt:
        robusta.batchnorm.adapt(model, adapt_type="batch_wise")
        for name,child in (model.backbone.named_children()):
            if isinstance(child, nn.BatchNorm2d):
                # print(name)
                for param in child.parameters():
                    param.requires_grad = True
            else:
                for param in child.parameters():
                    param.requires_grad = False 
    else:
        for param in model.backbone.parameters():
            param.requires_grad = False

    if not vc_flag:
        model.conv1o1.weight.requires_grad = False
    else:
        model.conv1o1.weight.requires_grad = True

    if not mix_flag:
        model.mix_model.requires_grad = False
    else:
        model.mix_model.requires_grad = True
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    #     else:
    #         print("no ", name )
    # exit()

    if use_gce:
        print("\nUsing GCE Loss\n")
    rpl_loss = robusta.selflearning.GeneralizedCrossEntropy(q=0.8)
    classification_loss = nn.CrossEntropyLoss()
    cluster_loss = ClusterLoss()

    optimizer = torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.98)

    print('Training')

    for epoch in range(epochs):
        # print("EPOCH {}".format(epoch))
        out_file = open(out_file_name, 'a')
        train_loss = 0.0
        correct = 0
        start = time.time()
        model.train()
        if not args.bn_adapt:
            model.backbone.eval()
        for index, data in enumerate(train_loader):
            if index % 500 == 0 and index != 0:
                end = time.time()
                print('Epoch{}: {}/{}, Acc: {}, Loss: {} Time:{}'.format(epoch + 1, index, total_train, correct.cpu().item() / index, train_loss.cpu().item() / index, (end-start)))
                start = time.time()

            input, _, label = data

            input = input.cuda(device_ids[0])
            label = label.cuda(device_ids[0])

            output, vgg_feat, like = model(input)

            out = output.argmax(1)
            correct += torch.sum(out == label)
            # if index==113: #/change shuffle in dataloader
            #     print("")
            if not use_gce:
                class_loss = classification_loss(output, label) / output.shape[0]
            else:
                class_loss = rpl_loss(output) / output.shape[0]
                # if torch.sum(torch.isnan(class_loss))>0 or torch.isnan(class_loss).any():
                #     print("uh")
                #     class_loss = rpl_loss(output)
                # class_loss /= output.shape[0]
            
            loss = class_loss
            if alpha != 0:
                clust_loss = cluster_loss(vgg_feat, model.conv1o1.weight) / output.shape[0]
                if torch.isnan(clust_loss):
                    clust_loss=0
                    print("{} :Cluster Loss is NaN".format(index))
                loss += alpha * clust_loss

            if beta!=0:
                mix_loss = like[0,label[0]] #/fix this
                loss += -beta *mix_loss

            # with torch.autograd.set_detect_anomaly(True):
            loss.backward()

            # pseudo batches
            if batch_size==1:
                if np.mod(index,pseudo_batch_size)==0 and index!=0:
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                optimizer.step()
                optimizer.zero_grad()
            # print(model.conv1o1.weight)

            train_loss += loss.detach() * input.shape[0]

        updated_clutter = update_clutter_model(model,device_ids)
        model.clutter_model = updated_clutter
        scheduler.step()
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
        out_file.close()
    return best_check

if __name__ == '__main__':

    if backbone_type=='vgg':
        if layer=='pool4':
            extractor = models.vgg16(pretrained=True).features[0:24]
        else:
            extractor = models.vgg16(pretrained=True).features
    elif backbone_type =='vgg_tr':
        """VGG model trained from scratch or pretrained"""
        print("Loading robin trained model")
        layer = 'pool5'  # 'pool5','pool4'
        # saved_model = 'baseline_models/train_None_lr_0.01_pascal3d+_pretrained_False_epochs_15_occ_False_backbonevgg_0/vgg14.pth'
        # saved_model = 'baseline_models/Robin-train-vgg_bn.pth' # adapted vgg_bn
        saved_model = 'baseline_models/pascal3d+None_lr_0.001_scratFalsepretrFalse_ep60_occFalse_backbvgg_bn_0/vgg_bn52.pth'
        if args.tr_model is not None:
            print("CAUTION: Loading argument model for backbone")
            # saved_model = 'baseline_models/snow_adapted_vgg_bn_pascal3d+.pth'
            saved_model = args.tr_model
        # saved_model = 'baseline_models/robinNone_lr_0.001_scratFalsepretrFalse_ep60_occFalse_backbvgg_bn_0/vgg_bn51.pth'
        load_dict = torch.load(saved_model, map_location='cuda:{}'.format(0))
        # tmp = models.vgg16(pretrained=False)
        tmp = models.vgg16_bn(pretrained=False)
        num_ftrs = tmp.classifier[6].in_features
        if dataset in ['robin', 'pseudorobin', 'occludedrobin']:
            tmp.classifier[6] = torch.nn.Linear(num_ftrs, 11)
        else:
            tmp.classifier[6] = torch.nn.Linear(num_ftrs, len(categories))
        tmp.load_state_dict(load_dict['state_dict'])
        tmp.eval()
        extractor = tmp.features
    elif backbone_type=='resnet50' or backbone_type=='resnext':
        extractor = resnet_feature_extractor(backbone_type, layer, load_bbone=args.tr_model)

    extractor.cuda(device_ids[0]).eval()
    if DA:
        weights = getVmfKernels(da_dict_dir, device_ids)
    else:
        weights = getVmfKernels(dict_dir, device_ids)

    if bool_load_pretrained_model:
        assert(args.ctrainpth is not None)
        # pretrained_file = 'PATH TO .PTH FILE HERE'
        pretrained_file = args.ctrainpth
    else:
        pretrained_file = ''


    occ_likely = []
    for i in range(len(categories_train)):
        # setting the same occlusion likelihood for all classes
        occ_likely.append(likely)

    if DA:
        mix_models = getCompositionModel(device_ids,da_mix_model_path,layer,categories_train,compnet_type=compnet_type)
    else:
        # load the CompNet initialized with ML and spectral clustering
        mix_models = getCompositionModel(device_ids,mix_model_path,layer,categories_train,compnet_type=compnet_type)
    net = Net(extractor, weights, vMF_kappa, occ_likely, mix_models, bool_mixture_bg=bool_mixture_model_bg,compnet_type=compnet_type,num_mixtures=num_mixtures, vc_thresholds=cfg.MODEL.VC_THRESHOLD)
    if bool_load_pretrained_model:
        net.load_state_dict(torch.load(pretrained_file, map_location='cuda:{}'.format(device_ids[0]))['state_dict'])

    net = net.cuda(device_ids[0])

    train_imgs=[]
    train_masks = []
    train_labels = []
    val_imgs = []
    val_labels = []
    val_masks=[]
    # DA = True #! remove!
    # get training and validation images
    for occ_level in occ_levels_train:
        if occ_level == 'ZERO':
            occ_types = ['']
            train_fac=0.9
        else:
            occ_types = ['_white', '_noise', '_texture', '']
            train_fac=0.1

        for occ_type in occ_types:
            if DA and mode not in ['mixed', 'corres']:
                if dataset in ['robin','pseudorobin']:
                    print("Loading Robin test data (pseudo {})".format(dataset=='pseudorobin'))
                    imgs, labels, masks = getImg('test', categories_train, dataset, data_path, categories, \
                        occ_level, occ_type, bool_load_occ_mask=False, subcat=cat)
                elif dataset in ['pascal3d+', 'pseudopascal']:
                    print("Loading Pascal3d+ test data (pseudo {})".format(dataset=='pseudopascal'))
                    imgs, labels, masks = getImg('test', categories, dataset, data_path, categories, \
                        occ_level, occ_type, bool_load_occ_mask=False, determinate=True, corruption=corr)
            else:
                print("Loading {} train data".format(dataset))
                imgs, labels, masks = getImg('train', categories_train, dataset, data_path, \
                    categories, occ_level, occ_type, bool_load_occ_mask=False, subcat=cat)
            nimgs=len(imgs)
            if not test_as_val:
                for i in range(nimgs):
                    if (random.randint(0, nimgs - 1) / nimgs) <= train_fac:
                        train_imgs.append(imgs[i])
                        train_labels.append(labels[i])
                        train_masks.append(masks[i])
                    elif not bool_train_with_occluders:
                        val_imgs.append(imgs[i])
                        val_labels.append(labels[i])
                        val_masks.append(masks[i])
            else:
                train_imgs, train_labels, train_masks = imgs, labels, masks
                print("Loading Test as Val of {}".format(dataset))
                if dataset=='robin':
                    # if dataset in ['robin','pseudorobin']:
                    val_imgs, val_labels, val_masks = getImg('test', categories_train, dataset, data_path, \
                        categories, occ_level, occ_type, bool_load_occ_mask=False, subcat=cat)
                elif dataset=='pascal3d+':
                    print("Loading Pascal3d+ test data (pseudo {})".format(dataset=='pseudopascal'))
                    val_imgs, val_labels, val_masks = getImg('test', categories, dataset, data_path, categories, \
                        occ_level, occ_type, bool_load_occ_mask=False, determinate=True, corruption=corr)
                else:
                    raise(RuntimeError)

    print('Total imgs for train ' + str(len(train_imgs)))
    print('Total imgs for val ' + str(len(val_imgs)))
    train_imgset = Imgset(train_imgs,train_masks, train_labels, imgLoader,bool_square_images=bool_square_images)

    val_imgsets = []
    if val_imgs:
        val_imgset = Imgset(val_imgs,val_masks, val_labels, imgLoader,bool_square_images=bool_square_images)
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

    train(model=net, train_data=train_imgset, val_data=val_imgsets, epochs=ncoord_it, batch_size=batch_size,
          learning_rate=lr, savedir=out_dir, alpha=alpha,beta=beta, vc_flag=vc_flag, mix_flag=mix_flag)


