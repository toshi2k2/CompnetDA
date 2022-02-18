"""Training VGGs"""
import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from Code.model import Net
from Code.helpers import getImg, Imgset, imgLoader, save_checkpoint
from Code.config import device_ids, categories, categories_train, dataset, data_path, \
	backbone_type, vggmodel_dir, layer
from Code.config import config as cfg
from torch.utils.data import DataLoader
# from Code.losses import ClusterLoss
# from Code.model import resnet_feature_extractor
import torchvision.models as models

import time
import torch
import torch.nn as nn
import numpy as np
import random

#---------------------
# Training Parameters
#---------------------
# beta = 3 # mix loss
# likely = 0.6 # occlusion likelihood
lr = 1e-3 # learning rate
batch_size = 64
# Training setup
# mix_flag = True # train mixture components
ncoord_it = 60 	#number of epochs to train
corr = None#'snow'
scratch = False#True

# bool_mixture_model_bg = False #True: use a mixture of background models per pixel, False: use one bg model for whole image
bool_load_pretrained_model = False#True
bool_train_with_occluders = False

if dataset == 'robin':
    categories_train.remove('bottle')
    categories = categories_train

if bool_train_with_occluders:
	occ_levels_train = ['ZERO', 'ONE', 'FIVE', 'NINE']
else:
	occ_levels_train = ['ZERO']

backbone_type = 'resnet50'

out_dir = vggmodel_dir + '/{}{}_lr_{}_scrat{}pretr{}_ep{}_occ{}_backb{}_{}/'.format(dataset,corr, lr, scratch,\
	bool_load_pretrained_model,ncoord_it,bool_train_with_occluders,backbone_type,device_ids[0])


def train(model, train_data, val_data, epochs, batch_size, learning_rate, savedir):
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
		val_loader = DataLoader(dataset=val_data[i], batch_size=1, shuffle=True)
		val_loaders.append(val_loader)

	classification_loss = nn.CrossEntropyLoss()

	# optimizer = torch.optim.Adagrad(params=filter(lambda param: param.requires_grad, model.parameters()), lr=learning_rate)
	# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.98)

	# Observe that all parameters are being optimized
	optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

	# Decay LR by a factor of 0.1 every 7 epochs
	# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=0.95)
	# batch_size=64
	print('Training')

	for epoch in range(epochs):
		out_file = open(out_file_name, 'a')
		train_loss = 0.0
		correct = 0
		start = time.time()
		model.train()
		for index, data in enumerate(train_loader):
			if index % 1000 == 0 and index != 0:
				end = time.time()
				print('Epoch{}: {}/{}, Acc: {}, Loss: {} Time:{}'.format(epoch + 1, index, total_train, correct.cpu().item() / index, train_loss.cpu().item() / index, (end-start)))
				start = time.time()

			input, _, label = data

			input = input.cuda(device_ids[0])
			label = label.cuda(device_ids[0])

			# optimizer.zero_grad()

			output = model(input)

			out = output.argmax(1)
			correct += torch.sum(out == label)
			class_loss = classification_loss(output, label) / output.shape[0]
			# class_loss = classification_loss(output, label)

			loss = class_loss
			# if alpha != 0:
			# 	clust_loss = cluster_loss(vgg_feat, model.conv1o1.weight) / output.shape[0]
			# 	loss += alpha * clust_loss

			# if beta!=0:
			# 	mix_loss = like[0,label[0]]
			# 	loss += -beta *mix_loss

			# with torch.autograd.set_detect_anomaly(True):
			loss.backward()

			# # pseudo batches
			# if np.mod(index,batch_size)==0 and index!=0:
			optimizer.step()
			optimizer.zero_grad()

			train_loss += loss.detach() * input.shape[0]
			# optimizer.step()
		# if epoch < 50:
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
					output = model(input)
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
				save_checkpoint(best_check, savedir + backbone_type + str(epoch + 1) + '.pth', True)

			print('\n')
		out_file.close()
	return best_check

if __name__ == '__main__':

	if backbone_type=='vgg':
		net = models.vgg16(pretrained=not scratch)
	# elif backbone_type=='resnet50' or backbone_type=='resnext':
	# 	net = resnet_feature_extractor(backbone_type, layer)
	elif backbone_type=='vgg_bn':
		net = models.vgg16_bn(pretrained=not scratch)
	elif backbone_type=='resnet50':# or backbone_type=='resnext':
		net = models.resnet50(pretrained=not scratch)
	else:
		raise(RuntimeError)
	# print(net)
	if backbone_type in ['vgg', 'vgg_bn']:
		num_ftrs = net.classifier[6].in_features
		net.classifier[6] = nn.Linear(num_ftrs, len(categories_train))
	elif backbone_type in ["resnet50"]:
		# num_ftrs = net.fc.in_features
		net.fc = nn.Linear(net.fc.in_features, len(categories_train))
	# print(net)

	net = net.cuda(device_ids[0])

	if bool_load_pretrained_model:
		pretrained_file = 'baseline_models/train_None_lr_0.01_pascal3d+_scratchTruepretrained_False_epochs_50_occ_False_backbonevgg_0/vgg13.pth'
	else:
		pretrained_file = ''

	if bool_load_pretrained_model:
		print("loading pre-trained model")
		net.load_state_dict(torch.load(pretrained_file, map_location='cuda:{}'.format(device_ids[0]))['state_dict'])

	# net = net.cuda(device_ids[0])

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
			if corr is not None:
				imgs, labels, masks = getImg('train', categories_train, dataset, data_path, categories, occ_level, \
					occ_type, bool_load_occ_mask=False, determinate=True, corruption=corr)
			else:
				print("USING CLEAN DATA\n")
				imgs, labels, masks = getImg('train', categories_train, dataset, data_path, categories, occ_level, \
					occ_type, bool_load_occ_mask=False)
				# imgs, labels, masks = getImg('test', categories_train, dataset, data_path, categories, occ_level, \
				# 	occ_type, bool_load_occ_mask=False)			
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
	train_imgset = Imgset(train_imgs,train_masks, train_labels, imgLoader,bool_square_images=True)

	val_imgsets = []
	if val_imgs:
		val_imgset = Imgset(val_imgs,val_masks, val_labels, imgLoader,bool_square_images=True)
		val_imgsets.append(val_imgset)

	# write parameter settings into output folder
	# load_flag = False
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	info = out_dir + 'config.txt'
	config_file = open(info, 'a')
	config_file.write(dataset)
	out_str = 'Train\nDir: {}, lr:{}\n'.format(out_dir, lr)
	config_file.write(out_str)
	print(out_str)
	out_str = 'pretrain{}_file{}'.format(bool_load_pretrained_model,pretrained_file)
	print(out_str)
	config_file.write(out_str)
	config_file.close()

	train(model=net, train_data=train_imgset, val_data=val_imgsets, epochs=ncoord_it, batch_size=batch_size,
		  learning_rate=lr, savedir=out_dir)


