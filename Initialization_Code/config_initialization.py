import os
import torchvision.models as models
from Code.model import resnet_feature_extractor
import torch

# Setup work
device_ids = [0]
data_path = 'data/'
model_save_dir = 'models/'

dataset = 'robin' # pascal3d+, coco, robin, pseudorobin
nn_type = 'resnet50' #vgg, vgg_bn, vgg_tr, resnet50, resnext, resnet152
vMF_kappa=10#30
vc_num = 512
vc_shape = 0#0

# categories = ['tvmonitor']
categories = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair', 'diningtable', 'motorbike', 'sofa',
			  'train', 'tvmonitor']
cat_test = ['aeroplane', 'bicycle', 'bus', 'car', 'motorbike', 'train']
robin_cats = ['context', 'weather', 'texture', 'pose', 'shape']

if nn_type =='vgg':
	layer = 'pool5'  # 'pool5','pool4'
	if layer == 'pool4':
		extractor=models.vgg16(pretrained=True).features[0:24]
	elif layer =='pool5':
		extractor = models.vgg16(pretrained=True).features
elif nn_type =='vgg_bn':
	layer = 'pool5'  # 'pool5','pool4'
	if layer == 'pool4':
		extractor=models.vgg16_bn(pretrained=True).features[0:24]
	elif layer =='pool5':
		extractor = models.vgg16_bn(pretrained=True).features
elif nn_type =='vgg_tr':
	"""VGG model trained from scratch"""
	layer = 'pool5'  # 'pool5','pool4'
	# saved_model = 'baseline_models/train_None_lr_0.01_pascal3d+_pretrained_False_epochs_15_occ_False_backbonevgg_0/vgg14.pth'
	# saved_model = 'baseline_models/snowadaptedvggbn.pth' # adapted vgg_bn
	# saved_model = 'baseline_models/None_adapted_vgg_bn_robin.pth' # adapted vgg_bn for robin
	saved_model = 'baseline_models/robinNone_lr_0.001_scratFalsepretrFalse_ep60_occFalse_backbvgg_bn_0/vgg_bn51.pth'
	load_dict = torch.load(saved_model, map_location='cuda:{}'.format(0))
	# tmp = models.vgg16(pretrained=False)
	tmp = models.vgg16_bn(pretrained=False)
	num_ftrs = tmp.classifier[6].in_features
	if dataset == 'robin':
		tmp.classifier[6] = torch.nn.Linear(num_ftrs, len(categories)-1)
	else:
		tmp.classifier[6] = torch.nn.Linear(num_ftrs, len(categories))
	tmp.load_state_dict(load_dict['state_dict'])
	tmp.eval()
	extractor = tmp.features
elif nn_type[:6]=='resnet' or nn_type=='resnext' or nn_type=='alexnet':
	layer = 'last' # 'last','second'
	extractor=resnet_feature_extractor(nn_type,layer)

extractor.cuda(device_ids[0]).eval()

init_path = model_save_dir+'init_{}/'.format(nn_type)
if not os.path.exists(init_path):
	os.makedirs(init_path)

dict_dir = init_path+'dictionary_{}/'.format(nn_type)
if not os.path.exists(dict_dir):
	os.makedirs(dict_dir)

sim_dir = init_path+'similarity_{}_{}_{}/'.format(nn_type,layer,dataset)

#* Dirs for domain adapted feature data
da_init_path = model_save_dir+'da_init_{}/'.format(nn_type)
if not os.path.exists(da_init_path):
	os.makedirs(da_init_path)

da_dict_dir = da_init_path+'dictionary_{}/'.format(nn_type)
if not os.path.exists(da_dict_dir):
	os.makedirs(da_dict_dir)

da_sim_dir = da_init_path+'similarity_{}_{}_{}/'.format(nn_type,layer,dataset)
#* ############################################

Astride_set = [2, 4, 8, 16, 32]  # stride size
featDim_set = [64, 128, 256, 512, 512]  # feature dimension
Arf_set = [6, 16, 44, 100, 212]  # receptive field size
Apad_set = [2, 6, 18, 42, 90]  # padding size

if layer =='pool4' or layer =='second':
	Astride = Astride_set[3]
	Arf = Arf_set[3]
	Apad = Apad_set[3]
	offset = 3
elif layer =='pool5' or layer == 'last':
	Astride = Astride_set[3]
	Arf = 170
	Apad = Apad_set[4]
	offset = 3
