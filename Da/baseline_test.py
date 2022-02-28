import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
import torch
import numpy as np
from torch.utils.data import DataLoader
from Code.config import categories, categories_train, dataset, data_path, \
	device_ids, model_save_dir, backbone_type
from Code.model import Net
from Code.helpers import getImg, Imgset, imgLoader, UnNormalize
import tqdm
import torchvision.models as models
from Initialization_Code.config_initialization import robin_cats

import torchvision
u = UnNormalize()

###################
# Test parameters #
###################
# saved_model = 'baseline_models/ROBIN-train-resnet50.pth'
backbone_type='vgg_bn'
# saved_model = 'baseline_models/train_None_lr_0.01_{}_scratchFalsespretrained_False_epochs_50_occ_False_backbone{}_0/{}3.pth'.format(dataset, backbone_type, backbone_type)
# saved_model = 'baseline_models/train_None_lr_0.0001_robin_scratchFalsepretrained_False_epochs_100_occ_False_backboneresnet50_0/resnet5047.pth'
# saved_model = 'baseline_models/train_None_lr_0.01_pascal3d+_scratchTruepretrained_True_epochs_50_occ_False_backbonevgg_0/vgg1.pth'
# saved_model = 'baseline_models/train_None_lr_0.01_robin_scratchFalsepretrained_False_epochs_50_occ_False_backbonevgg_bn_0/vgg_bn3.pth'
saved_model = '/mnt/sda1/cl_reps/Robin/vggbnshape_bn/best.pth'
# saved_model = 'baseline_models/ROBIN-train-resnet50.pth'
# saved_model = 'baseline_models/robinNone_lr_0.001_scratFalsepretrFalse_ep60_occFalse_backbvgg_bn_0/vgg_bn51.pth'
corr = None#'snow'
# backbone_type = 'resnet50'
bool_square_images=True
dataset = 'occludedrobin'

likely = 0.6  # occlusion likelihood
occ_levels = ['ZERO','ONE', 'FIVE', 'NINE'] # occlusion levels to be evaluated [0%,20-40%,40-60%,60-80%]

if dataset in ['robin', 'psuedorobin', 'occludedrobin']:
	# occ_levels = ['ZERO']
	categories_train.remove('bottle')
	categories = categories_train
	# robin_cats = [''] # option will run on entire robin testset
	robin_cats = ['shape']

if dataset in ['robin', 'psuedorobin']:
	occ_levels = ['ZERO']

def test(models, test_data, batch_size):
	test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
	print('Testing')
	nclasses = 12 # models[0].num_classes
	correct = np.zeros(nclasses)
	total_samples = np.zeros(nclasses)
	scores = np.zeros((0,nclasses))

	with torch.no_grad():
		for i, data in enumerate(tqdm.tqdm(test_loader)):
			# print(data[0].shape)
			input,mask, label = data
			input.requires_grad = False

			# saving images
			if i%1000==0:
				im = torchvision.transforms.functional.to_pil_image(u(input[0]))
				# print(type(im))
				# im.show()
				im.save('results/btest_snow_%s.png' % i)
				del im

			if device_ids:
				input = input.cuda(device_ids[0])
			c_label = label.numpy()

			output, *_ = models(input)
			out = output.cpu().numpy()
			# print(out, c_label, out.argmax())
			# scores = np.concatenate((scores,out))
			out = out.argmax()
			# out = out[:11].argmax()
			# if out>2:
			# 	out-=1
			correct[c_label] += np.sum(out == c_label)

			total_samples[c_label] += 1

	for i in range(nclasses):
		if total_samples[i]>0:
			print('Class {}: {:1.3f}'.format(categories_train[i],correct[i]/total_samples[i]))
	test_acc = (np.sum(correct)/np.sum(total_samples))
	return test_acc, scores

if __name__ == '__main__':

	if saved_model is None:
		path = model_save_dir + 'best.pth'
	else:
		path = saved_model

	occ_likely = []
	for _ in range(len(categories_train)):
		occ_likely.append(likely)

	############################
	# Saved Model
	############################
	if backbone_type=='vgg':
		model = models.vgg16(pretrained=True)
		model.classifier[6]  = torch.nn.Linear(4096, 12)
	elif backbone_type=='vgg_bn':
		model = models.vgg16_bn(pretrained=True)
		model.classifier[6]  = torch.nn.Linear(4096, len(categories_train))
	elif backbone_type == 'resnet18' or backbone_type == 'resnext' or \
		backbone_type=='densenet':
		exit("baselines {} not implemented".format(backbone_type))
	elif backbone_type=='resnet50':# or backbone_type=='resnext':
		model = models.resnet50(pretrained=True)
		model.fc = torch.nn.Linear(model.fc.in_features, len(categories_train))
		# model.fc = torch.nn.Linear(model.fc.in_features, 12)

	if device_ids:
		load_dict = torch.load(path, map_location='cuda:{}'.format(0))
	else:
		load_dict = torch.load(path, map_location='cpu')

	model.load_state_dict(load_dict['state_dict'])

	model.cuda().eval() #* IS THIS NEEDED?

	############################
	# Test Loop
	############################
	for occ_level in occ_levels:

		if occ_level == 'ZERO':
			occ_types = ['']
		else:
			if dataset=='pascal3d+':
				occ_types = ['']#['_white','_noise', '_texture', '']
			elif dataset in ['coco','occludedrobin']:
				occ_types = ['']

		for index, occ_type in enumerate(occ_types):
			for cat in robin_cats:
				if cat == '' and dataset in ['robin','occludedrobin']: 
					cat = None
					print("All Subcategories\n")
				elif dataset in ['robin','occludedrobin']:
					print(cat)
					cat=[cat]
				# load images
				if corr is not None:
					print("Loading corrupted data\n")
					test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
						categories, occ_level, occ_type,bool_load_occ_mask=True, determinate=True, corruption=corr)
				else:
					test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
						categories, occ_level, occ_type,bool_load_occ_mask=True, subcat=cat)

				#* removing an image due to errors
				if corr is not None:
					errs = ["data/pascal3d+_occ_{}/carLEVEL{}/n03770679_14513_2.JPEG".format(corr, occ_level)]
					# errs = ['data/pascal3d+_occ_snow/carLEVELONE/n03770679_14513_2.JPEG', 'data/pascal3d+_occ_snow/carLEVELFIVE/n03770679_14513_2.JPEG', 'data/pascal3d+_occ_snow/carLEVELNINE/n03770679_14513_2.JPEG']
					for es in errs:
						if es in test_imgs:
							idx_rm = test_imgs.index(es)
							del test_imgs[idx_rm]
							del test_labels[idx_rm]
							del masks[idx_rm]
				#*

				print('Total imgs for test of occ_level {} and occ_type {} '.format(occ_level, occ_type) + str(len(test_imgs)))
				"""test_imgs is list of image path and name strings"""
				# get image loader
				test_imgset = Imgset(test_imgs, masks, test_labels, imgLoader, bool_square_images=bool_square_images)
				# compute test accuracy
				acc, scores = test(models=model, test_data=test_imgset, batch_size=1)
				out_str = 'Model Name: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc)
				print(out_str)
