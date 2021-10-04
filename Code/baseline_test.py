import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
import torch
import numpy as np
from torch.utils.data import DataLoader
from config import categories, categories_train, dataset, data_path, \
	device_ids, model_save_dir, backbone_type
from model import Net
from helpers import getImg, Imgset, imgLoader, UnNormalize
import tqdm
import torchvision.models as models

import torchvision
u = UnNormalize()

###################
# Test parameters #
###################
likely = 0.6  # occlusion likelihood
occ_levels = ['ZERO', 'ONE', 'FIVE', 'NINE'] # occlusion levels to be evaluated [0%,20-40%,40-60%,60-80%]

def test(models, test_data, batch_size):
	test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
	print('Testing')
	nclasses = 12 # models[0].num_classes
	correct = np.zeros(nclasses)
	total_samples = np.zeros(nclasses)
	scores = np.zeros((0,nclasses))

	with torch.no_grad():
		for i, data in enumerate(tqdm.tqdm(test_loader)):
			input,mask, label = data
			input.requires_grad = False

			# saving images
			if i%500==0:
				im = torchvision.transforms.functional.to_pil_image(u(input[0]))
				# print(type(im))
				# im.show()
				im.save('results/4checksnow_%s.png' % i)
				del im

			if device_ids:
				input = input.cuda(device_ids[0])
			c_label = label.numpy()

			output, *_ = models(input)
			out = output.cpu().numpy()
			# print(out, c_label, out.argmax())
			# scores = np.concatenate((scores,out))
			out = out.argmax()
			correct[c_label] += np.sum(out == c_label)

			total_samples[c_label] += 1

	for i in range(nclasses):
		if total_samples[i]>0:
			print('Class {}: {:1.3f}'.format(categories_train[i],correct[i]/total_samples[i]))
	test_acc = (np.sum(correct)/np.sum(total_samples))
	return test_acc, scores

if __name__ == '__main__':

	path = model_save_dir + 'best.pth'

	occ_likely = []
	for _ in range(len(categories_train)):
		occ_likely.append(likely)

	############################
	# Saved Model
	############################
	if backbone_type=='vgg':
		pass
	elif backbone_type=='resnet50' or backbone_type == 'resnet18' or backbone_type == 'resnext' or backbone_type=='densenet':
		exit("baselines {} not implemented".format(backbone_type))

	model = models.vgg16(pretrained=True)
	model.classifier[6]  = torch.nn.Linear(4096, 12)

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
			elif dataset=='coco':
				occ_types = ['']

		for index, occ_type in enumerate(occ_types):
			# load images
			test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
				categories, occ_level, occ_type,bool_load_occ_mask=True, determinate=True, corruption='snow')
			#* removing an image due to errors
			errs = ['data/pascal3d+_occ_snow/carLEVELONE/n03770679_14513_2.JPEG', 'data/pascal3d+_occ_snow/carLEVELFIVE/n03770679_14513_2.JPEG', 'data/pascal3d+_occ_snow/carLEVELNINE/n03770679_14513_2.JPEG']
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
			test_imgset = Imgset(test_imgs, masks, test_labels, imgLoader, bool_square_images=False)
			# compute test accuracy
			acc, scores = test(models=model, test_data=test_imgset, batch_size=1)
			out_str = 'Model Name: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc)
			print(out_str)
