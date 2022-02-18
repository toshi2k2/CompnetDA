from genericpath import exists
import os
import torch
import cv2
import glob
import torch.nn.functional as F
from Code.config import vc_num, categories, occ_types_vmf, occ_types_bern
from Code.vMFMM import *
from torchvision import transforms
from PIL import Image

from imagecorruptions import corrupt
import numpy as np
from random import randrange


def update_clutter_model(net,device_ids,compnet_type='vmf'):
	idir = 'background_images/';
	updated_models = torch.zeros((0,vc_num))
	if device_ids:
		updated_models = updated_models.cuda(device_ids[0])

	if compnet_type=='vmf':
		occ_types=occ_types_vmf
	elif compnet_type=='bernoulli':
		occ_types=occ_types_bern

	for j in range(len(occ_types)):
		occ_type = occ_types[j]
		with torch.no_grad():
			files = glob.glob(idir + '*'+occ_type+'.JPEG')
			clutter_feats = torch.zeros((0,vc_num))
			if device_ids:
				clutter_feats=clutter_feats.cuda(device_ids[0])
			for i in range(len(files)):
				file = files[i]
				img,_ = imgLoader(file,[[]], bool_resize_images=False,bool_square_images=False)
				if device_ids:
					img =img.cuda(device_ids[0])

				feats 	 = net.activation_layer(net.conv1o1(net.backbone(img.reshape(1,img.shape[0],img.shape[1],img.shape[2]))))[0].transpose(1,2)
				feats_reshape = torch.reshape(feats, [vc_num, -1]).transpose(0,1)
				clutter_feats = torch.cat((clutter_feats, feats_reshape))

			mean_activation = torch.reshape(torch.sum(clutter_feats,dim=1),(-1,1)).repeat([1,vc_num])#clutter_feats.sum().reshape(-1,1).numpy().repeat(512,axis=1)#torch.reshape(torch.sum(clutter_feats,axis=1),(-1,1)).repeat(512,axis=1)
			if compnet_type=='bernoulli':
				boo = torch.sum(mean_activation, dim=1) != 0
				mean_vec = torch.mean(clutter_feats[boo]/mean_activation[boo], dim=0)
				updated_models = torch.cat((updated_models, mean_vec.reshape(1, -1)))
			else:
				if occ_type== '_white' or occ_type== '_noise':
					mean_vec = torch.mean(clutter_feats/mean_activation, dim=0)  # F.normalize(torch.mean(clutter_feats,dim=0),p=1,dim=0)#
					updated_models = torch.cat((updated_models, mean_vec.reshape(1, -1)))
				else:
					nc = 5
					model = vMFMM(nc, 'k++')
					model.fit(clutter_feats.cpu().numpy(), 30.0, max_it=150)
					mean_vec = torch.zeros(nc,clutter_feats.shape[1]).cuda(device_ids[0])
					clust_cnt = torch.zeros(nc)
					for v in range(model.p.shape[0]):
						assign = np.argmax(model.p[v])
						mean_vec[assign] += clutter_feats[v]
						clust_cnt[assign]+=1
					mean_vec = (mean_vec.t()/clust_cnt.cuda(device_ids[0])).t()
					updated_models = torch.cat((updated_models, mean_vec))



	return updated_models

def getVmfKernels(dict_dir, device_id):
	vc = np.load(dict_dir, allow_pickle=True)
	vc = vc[:, :, np.newaxis, np.newaxis]
	vc = torch.from_numpy(vc).type(torch.FloatTensor)
	if device_id:
		vc = vc.cuda(device_id[0])
	return vc

def getCompositionModel(device_id,mix_model_path,layer,categories,compnet_type='vmf',num_mixtures=4):

	mix_models = []
	msz = []
	for i in range(len(categories)):
		filename = mix_model_path + '/mmodel_' + categories[i] + '_K{}_FEATDIM{}_{}_specific_view.pickle'.format(num_mixtures, vc_num, layer)
		mix = np.load(filename, allow_pickle=True)
		if compnet_type=='vmf':
			mix = np.array(mix)
		elif compnet_type == 'bernoulli':
			mix = np.array(mix[0])
		mix = np.transpose(mix, [0, 1, 3, 2])
		mix_models.append(torch.from_numpy(mix).type(torch.FloatTensor))
		msz.append(mix.shape)

	maxsz = np.max(np.asarray(msz),0)
	maxsz[2:4] = maxsz[2:4] + (np.mod(maxsz[2:4], 2) == 0)
	if layer == 'pool4' and compnet_type=='vmf':
		# Need to cut down the model to enable training
		maxsz[2] = maxsz[2] - 20#42
		maxsz[3] = maxsz[3] - 40#92

	mm = torch.zeros(0,vc_num,maxsz[2],maxsz[3])
	for i in range(len(categories)):
		mix = mix_models[i]
		cm, hm, wm = mix.shape[1:]
		# pad height
		if layer =='pool5':
			# because feature height is odd (7) copmared to 14 in pool4
			diff1 = int(np.ceil((maxsz[2] - hm) / 2))
		else:
			diff1 = int(np.floor((maxsz[2] - hm) / 2))
		diff2 = maxsz[2] - hm - diff1
		if diff1 < 0 or diff2<0:
			mix = mix[:,:,np.abs(diff1):np.abs(diff1)+maxsz[2]]
		else:
			if compnet_type=='vmf':
				mix = F.pad(mix, (0, 0, diff1, diff2, 0, 0, 0, 0), 'constant', 0)
			elif compnet_type=='bernoulli':
				mix = F.pad(mix, (0, 0, diff1, diff2, 0, 0, 0, 0), 'constant', np.log(1 / (1 - 1e-3)))
		# pad width
		if layer =='pool5':
			# because feature height is odd (7) copmared to 14 in pool4
			diff1 = int(np.ceil((maxsz[3] - wm) / 2))
		else:
			diff1 = int(np.floor((maxsz[3] - wm) / 2))
		diff2 = maxsz[3] - wm - diff1
		if diff1 < 0 or diff2<0:
			mix = mix[:, :, :, np.abs(diff1):np.abs(diff1) + maxsz[3]]
		else:
			if compnet_type=='vmf':
				mix = F.pad(mix, (diff1, diff2, 0, 0, 0, 0, 0, 0), 'constant', 0)
			elif compnet_type == 'bernoulli':
				mix = F.pad(mix, (diff1, diff2, 0, 0, 0, 0, 0, 0), 'constant', np.log(1 / (1 - 1e-3)))
		mm = torch.cat((mm,mix),dim=0)
	if device_id:
		mm = mm.cuda(device_id[0])
	return mm

def pad_to_size(x, to_size):
	padding = [(to_size[1] - x.shape[3]) // 2, (to_size[1] - x.shape[3]) - (to_size[1] - x.shape[3]) // 2, (to_size[0] - x.shape[2]) // 2, (to_size[0] - x.shape[2]) - (to_size[0] - x.shape[2]) // 2]
	return F.pad(x, padding)

def myresize(img, dim, tp):
	H, W = img.shape[0:2]
	if tp == 'short':
		if H <= W:
			ratio = dim / float(H)
		else:
			ratio = dim / float(W)

	elif tp == 'long':
		if H <= W:
			ratio = dim / float(W)
		else:
			ratio = dim / float(H)

	return cv2.resize(img, (0, 0), fx=ratio, fy=ratio)

def getImg(mode,categories, dataset, data_path, cat_test=None, occ_level='ZERO', occ_type=None, \
	bool_load_occ_mask = False, determinate=False, corruption=None, corr_bck=None, subcat=None): 
	#* determinate = True # for using fixed corrupted dataset - only for pascal3d and no occlusion
	#* corruption = type of corruption used - cannot be one for determinate = True
	#* subcat = subcategory filter in robin dataset
	assert(determinate and (corruption is not None) or not determinate and (corruption is None))

	if mode == 'train':
		train_imgs = []
		train_labels = []
		train_masks = []
		for category in categories:
			if dataset == 'pascal3d+':
				if occ_level == 'ZERO':
					filelist = data_path + 'pascal3d+_occ/' + category + '_imagenet_train' + '.txt'
					img_dir = data_path + 'pascal3d+_occ/TRAINING_DATA/' + category + '_imagenet'
				if determinate is True:
					if corr_bck is None:
						corr_img_dir = data_path + 'pascal3d+_occ_' + corruption + '/TRAINING_DATA/' + category + '_imagenet'
						if (not os.path.exists(corr_img_dir) or len(os.listdir(corr_img_dir))==0):
							corrupt_img_create(img_dir, corr_img_dir, corruption)
					else:
						if  occ_type=='':
							occ_mask_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level+'_mask_object'
						else:
							occ_mask_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level+'_mask'
						occ_mask_dir_obj = data_path + 'pascal3d+_occ/0_old_masks/'+category+'_imagenet_occludee_mask/'
						corr_img_dir = data_path + 'pascal3d+_occ_' + corruption + '-' + corr_bck + '/TRAINING_DATA/' + category + '_imagenet'
						if (not os.path.exists(corr_img_dir) or len(os.listdir(corr_img_dir))==0):
							masked_corrupt_img_create(img_dir, corr_img_dir, occ_mask_dir, occ_mask_dir_obj, occ_level, corruption, corr_bck)
					img_dir = corr_img_dir
			elif dataset == 'coco':
				if occ_level == 'ZERO':
					img_dir = data_path +'coco_occ/{}_zero'.format(category)
					filelist = data_path +'coco_occ/{}_{}_train.txt'.format(category, occ_level)
			elif dataset == 'robin':
				img_dir = './' + data_path + 'Robin/cls_train/{}/'.format(category)
				img_list = os.listdir(img_dir)
				img_list = [img_dir + s for s in img_list]
				train_imgs += img_list
				label = categories.index(category)
				train_labels += [label]*len(img_list)
				train_masks += [False,False]*len(img_list)

			if dataset in ['pascal3d+','coco']:
				with open(filelist, 'r') as fh:
					contents = fh.readlines()
				fh.close()
				img_list = [cc.strip() for cc in contents]
				label = categories.index(category)
				for img_path in img_list:
					if dataset=='coco':
						if occ_level == 'ZERO':
							img = img_dir + '/' + img_path + '.jpg'
						else:
							img = img_dir + '/' + img_path + '.JPEG'
					else:
						img = img_dir + '/' + img_path + '.JPEG'
					occ_img1 = []
					occ_img2 = []
					train_imgs.append(img)
					train_labels.append(label)
					train_masks.append([occ_img1,occ_img2])

		return train_imgs, train_labels, train_masks

	else:
		"""Test-time"""
		test_imgs = []
		test_labels = []
		occ_imgs = []
		for category in cat_test:
			if dataset == 'pascal3d+':
				filelist = data_path + 'pascal3d+_occ/' + category + '_imagenet_occ.txt'
				img_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level
				
				if determinate is True:
					corr_img_dir = data_path + 'pascal3d+_occ_' + corruption + '/' + category + 'LEVEL' + occ_level
					if (not os.path.exists(corr_img_dir) or len(os.listdir(corr_img_dir))==0):
						corrupt_img_create(img_dir, corr_img_dir, corruption)
					img_dir = corr_img_dir

				if bool_load_occ_mask:
					if  occ_type=='':
						occ_mask_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level+'_mask_object'
					else:
						occ_mask_dir = data_path + 'pascal3d+_occ/' + category + 'LEVEL' + occ_level+'_mask'
					occ_mask_dir_obj = data_path + 'pascal3d+_occ/0_old_masks/'+category+'_imagenet_occludee_mask/'
			elif dataset == 'coco':
				if occ_level == 'ZERO':
					img_dir = data_path+'coco_occ/{}_zero'.format(category)
					filelist = data_path+'coco_occ/{}_{}_test.txt'.format(category, occ_level)
				else:
					img_dir = data_path+'coco_occ/{}_occ'.format(category)
					filelist = data_path+'coco_occ/{}_{}.txt'.format(category, occ_level)
			elif dataset == 'robin':
				# print("loading robin test data")
				cats = ['context', 'weather', 'texture', 'pose', 'shape']
				if subcat is None:
					subcat = cats
				else: 
					assert(all(item in cats for item in subcat))
					# t = []
					# t.append(subcat)
					# subcat = t
				for sc in subcat:
					img_dir = './' + data_path + 'Robin/cls_test/{}/{}/'.format(category, sc)
					img_list = os.listdir(img_dir)
					img_list = [img_dir + s for s in img_list]
					test_imgs += img_list
					label = categories.index(category)
					test_labels += [label]*len(img_list)
					occ_imgs += [False,False]*len(img_list)
				#! add subcategory labels to output 
			elif dataset == 'pseudorobin':
				#! Needs varaible for data path
				img_dir = './' + data_path + 'Robin/cls_pseudo_test_all/{}/'.format(category)
				img_list = os.listdir(img_dir)
				img_list = [img_dir + s for s in img_list]
				test_imgs += img_list
				label = categories.index(category)
				test_labels += [label]*len(img_list)
				occ_imgs += [False,False]*len(img_list)

			if dataset in ['pascal3d+','coco']:
				if os.path.exists(filelist):
					with open(filelist, 'r') as fh:
						contents = fh.readlines()
					fh.close()
					img_list = [cc.strip() for cc in contents]
					label = categories.index(category)
					for img_path in img_list:
						if dataset != 'coco':
							if occ_level=='ZERO':
								img = img_dir + occ_type + '/' + img_path[:-2] + '.JPEG'
								occ_img1 = []
								# occ_img2 = []
								occ_img2 = occ_mask_dir_obj + '/' + img_path + '.png'
							else:
								img = img_dir + occ_type + '/' + img_path + '.JPEG'
								if bool_load_occ_mask:
									occ_img1 = occ_mask_dir + '/' + img_path + '.JPEG'
									occ_img2 = occ_mask_dir_obj + '/' + img_path + '.png'
								else:
									occ_img1 = []
									occ_img2 = []

						else:
							img = img_dir + occ_type + '/' + img_path + '.jpg'
							occ_img1 = []
							occ_img2 = []

						test_imgs.append(img)
						test_labels.append(label)
						occ_imgs.append([occ_img1,occ_img2])
				else:
					print('FILELIST NOT FOUND: {}'.format(filelist))
		return test_imgs, test_labels, occ_imgs

"""Image corruptions should be added here"""
def imgLoader(img_path,mask_path,bool_resize_images=True,bool_square_images=False, determinate=True):
	# determinate = False - do corruptions on images on the fly - adds randomness

	no_occluder = False  # if there is no mask for artificial occluders
	input_image = Image.open(img_path)
	if bool_resize_images:
		if bool_square_images:
			input_image=input_image.resize((224,224),Image.ANTIALIAS)
			# print(input_image.size)
		else:
			sz=input_image.size
			min_size = np.min(sz)
			if min_size!=224:
				input_image = input_image.resize((np.asarray(sz) * (224 / min_size)).astype(int),Image.ANTIALIAS)

	try:
		if mask_path[0]:
			mask1 = cv2.imread(mask_path[0])[:, :, 0]
			mask1 = myresize(mask1, 224, 'short')
			try:
				mask2 = cv2.imread(mask_path[1])[:, :, 0]
				mask2 = mask2[:mask1.shape[0], :mask1.shape[1]]
			except:
				mask = mask1
			try:
				mask = ((mask1 == 255) * (mask2 == 255)).astype(np.float)
			except:
				mask = mask1
		else:
			# mask = np.ones((img.shape[0], img.shape[1])) * 255.0
			no_occluder = True
			mask = np.ones((np.array(input_image).shape[0], np.array(input_image).shape[1])) * 255.0
			# print("ones", mask_path, len(mask_path))
	except TypeError as e:
		if mask_path == False:
			mask = np.ones((np.array(input_image).shape[0], np.array(input_image).shape[1])) * 255.0

	if determinate == False:
		"""Image Corruptions w/t object mask"""
		# input_image = Image.fromarray(corrupt(np.array(input_image), \
		# 	corruption_name='snow', severity=4))
		# input_image = Image.fromarray(corrupt(np.array(input_image), \
		# 	corruption_number=randrange(15), severity=randrange(5)+1))

		# input_image.show()
		"""Image Corruption with mask"""
		if len(mask_path)>1 and mask_path[1]:
			try:
				class_mask_temp = cv2.imread(mask_path[1])[:, :, 0]
				class_mask = class_mask_temp
			except TypeError:
				class_mask = mask
			# print("pre: ", class_mask.shape)
			class_mask = myresize(class_mask, 224, 'short')
			
			# print("short mask resize: ", class_mask.shape)
			if class_mask.shape[0]!= np.array(input_image).shape[0] or class_mask.shape[1]!= np.array(input_image).shape[1]:
				class_mask = cv2.resize(class_mask, (np.array(input_image).shape[1], np.array(input_image).shape[0]))
				# class_mask = cv2.resize(class_mask, (np.array(input_image).shape[1], 224))
			# if class_mask.shape[1]!= np.array(input_image).shape[1]:
			# 	class_mask = cv2.resize(class_mask, (224, np.array(input_image).shape[1]))
				# class_mask = myresize(class_mask, np.array(input_image).shape[1], 'long')
			# print("post resize mask: ", class_mask.shape, "image shape", np.array(input_image).shape)
			# input()
			
			############# Specific Corruptions ##############################
			corrupt_im = corrupt(np.array(input_image), corruption_name='pixelate', severity=4)
			########## random corruptions ###########################
			# corrupt_im = corrupt(np.array(input_image), corruption_number=randrange(15), severity=randrange(5)+1)
			#################################################################
			########Combining object and occluder mask filtering for corruption#########################
			# if not no_occluder:
			# 	try:
			# 		class_mask = class_mask + mask
			# 	except ValueError as e:
			# 		mask = cv2.resize(mask, (class_mask.shape[1], class_mask.shape[0]))
			# 		class_mask = class_mask + mask
			# 	class_mask[class_mask>=200] = 255
			#################################################################

				# l = Image.fromarray(class_mask)
				# l.show()
				# input()

			try:
				corrupt_im = corrupt_im.copy()
				# corrupt_im[class_mask==255] = 0
				corrupt_im[class_mask!=255] = 0
			except IndexError:
				print("IndexError:", np.array(input_image).shape, class_mask.shape)
			except ValueError as e:
				print(e)
				# print(corrupt_im.flags)
				# corrupt_im.setflags(write=1)
				# corrupt_im = corrupt_im.copy()
				# corrupt_im[class_mask==255] = 0
			for_gnd = np.array(input_image)
			# for_gnd[class_mask!=255] = 0
			for_gnd[class_mask!=255] = 1
			input_image = Image.fromarray(corrupt_im+for_gnd)
			# input_image.show()
			# input()

	mask = torch.from_numpy(mask)

	preprocess =  transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	img = preprocess(input_image)
	# print(img.shape)
	return img, mask

class Imgset():
	def __init__(self, imgs, masks, labels, loader,bool_square_images=False):
		self.images = imgs
		self.masks 	= masks
		self.labels = labels
		self.loader = loader
		self.bool_square_images = bool_square_images

	def __getitem__(self, index):
		fn = self.images[index]
		label = self.labels[index]
		mask = self.masks[index]
		img,mask = self.loader(fn,mask,bool_resize_images=True,bool_square_images=self.bool_square_images)
		return img, mask, label

	def __len__(self):
		return len(self.images)

def save_checkpoint(state, filename, is_best):
	if is_best:
		print("=> Saving new checkpoint")
		torch.save(state, filename)
	else:
		print("=> Validation Accuracy did not improve")


class UnNormalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def corrupt_img_create(img_dir, corrupted_dir, corruption):
	print("creating corrupted image dataset for {}".format(corruption))
	os.makedirs(corrupted_dir, exist_ok=True)
	print(img_dir)
	for file in os.listdir(img_dir):
		# print(file)
		if file.endswith(".JPEG"):
			filepath = os.path.join(img_dir, file)
			input_image = Image.open(filepath)
			try:
				corrupt_im = corrupt(np.array(input_image), corruption_name=corruption, severity=4)
			except AttributeError as e:
				print(e, file, img_dir)
				continue
			fullOutPath = os.path.join(corrupted_dir, file)
			Image.fromarray(corrupt_im).save(fullOutPath)
	return

def masked_corrupt_img_create(img_dir, corrupted_dir, occ_mask_dir, occ_mask_dir_obj, occ_level, for_corr, bck_corr):
	print("creating corrupted image dataset for foreground {} - bckgnd {}".format(for_corr, bck_corr))
	os.makedirs(corrupted_dir, exist_ok=True)
	print(img_dir)
	for file in os.listdir(img_dir):
		# print(file)
		if file.endswith(".JPEG"):
			filepath = os.path.join(img_dir, file)
			input_image = Image.open(filepath)
			# if occ_level=='ZERO':
			# 	occ_img1 = []
			# 	occ_img2 = occ_mask_dir_obj + '/' + file[:-5] + '.png'
			# else:
			# 	occ_img1 = occ_mask_dir + '/' + file
			# 	occ_img2 = occ_mask_dir_obj + '/' + file[:-5] + '.png'
			try:
				if occ_level != 'ZERO':
					occ_image1 = Image.open(os.path.join(occ_mask_dir, file))
				occ_image2 = Image.open(os.path.join(occ_mask_dir_obj, file[:-5] + '.png'))
				# print(file)
			except FileNotFoundError as e:
				print(e, file, img_dir)
				continue
			try:
				corrupt_im = corrupt(np.array(input_image), corruption_name=for_corr, severity=4)
			except AttributeError as e:
				print(e, file, img_dir)
				continue
			fullOutPath = os.path.join(corrupted_dir, file)
			Image.fromarray(corrupt_im).save(fullOutPath)
	return