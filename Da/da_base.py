"""Implementing two papers - batchnorm adapt and self training as baselines"""
import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
import torch
import numpy as np
from torch.utils.data import DataLoader
from Code.config import categories, categories_train, dataset, data_path, \
    device_ids, model_save_dir, backbone_type
from Code.model import Net
from Code.helpers import getImg, Imgset, imgLoader, UnNormalize,save_checkpoint
import tqdm
import torchvision.models as models
import robusta, copy
from Initialization_Code.config_initialization import robin_cats

import torchvision
u = UnNormalize()

###################
# Test parameters #
###################
# saved_model = 'baseline_models/ROBIN-train-resnet50.pth'
saved_model = 'baseline_models/robinNone_lr_0.001_scratFalsepretrFalse_ep60_occFalse_backbvgg_bn_0/vgg_bn50.pth'
# saved_model = 'baseline_models/train_None_lr_0.01_pascal3d+_pretrained_False_epochs_15_occ_False_backbonevgg_bn_0/vgg_bn12.pth'
# saved_model = 'models_old/best.pth'
# saved_model = 'baseline_models/train_None_lr_0.01_robin_scratchFalsepretrained_False_epochs_50_occ_False_backbonevgg_bn_0/vgg_bn3.pth'
# saved_model = 'baseline_models/train_None_lr_0.01_pascal3d+_pretrained_False_epochs_15_occ_False_backbonevgg_0/vgg14.pth'
# saved_model = 'baseline_models/train_None_lr_0.01_pascal3d+_scratchTruepretrained_True_epochs_50_occ_False_backbonevgg_0/vgg1.pth'
corr = None#'snow'
rbsta=True
backbone_type = 'vgg_bn' #vgg_bn, resnet50
save_checkpt = False
robin_all = False # run on all robin test nuisance altogether
check_changed_pmtr = False # check what parameters are training - run at least once for debugging

likely = 0.6  # occlusion likelihood
occ_levels = ['ZERO', 'ONE', 'FIVE', 'NINE'] # occlusion levels to be evaluated [0%,20-40%,40-60%,60-80%]

bool_train_with_occluders = False
if bool_train_with_occluders:
    occ_levels_train = ['ZERO', 'ONE', 'FIVE', 'NINE']
else:
    occ_levels_train = ['ZERO']
if dataset == 'robin':
	occ_levels = ['ZERO']
	categories_train.remove('bottle')
	categories = categories_train
else: robin_cats = ['']
if robin_all:
    robin_cats = ['']

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
			# if i%1000==0:
			# 	im = torchvision.transforms.functional.to_pil_image(u(input[0]))
				# print(type(im))
				# im.show()
				# im.save('results/btest_snow_%s.png' % i)
				# del im

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


def adapt(models, test_data, batch_size, robust_da, epochs=10, lr=1e-6):
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)
    print('Adapting')
    nclasses = 12 # models[0].num_classes
    correct = np.zeros(nclasses)
    total_samples = np.zeros(nclasses)
    scores = np.zeros((0,nclasses))

    o_model = copy.deepcopy(models)

    if robust_da:
        #Batch Norm adapt
        robusta.batchnorm.adapt(models, adapt_type="batch_wise")
        # self Learning
        parameters = robusta.selflearning.adapt(models, adapt_type="affine")
        optimizer = torch.optim.SGD(parameters, lr=lr)
        # print(models)
        # exit()
        # optimizer = torch.optim.SGD(models.parameters(), lr=0.001)
        rpl_loss = robusta.selflearning.GeneralizedCrossEntropy(q=0.8)
        models.train()
    
    for epoch in range(epochs):
        # with torch.no_grad():
        correct = np.zeros(nclasses)
        total_samples = np.zeros(nclasses)
        t_loss = 0.0
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            input,mask, label = data
            input.requires_grad = False

            # saving images
            # if i%1000==0:
            # 	im = torchvision.transforms.functional.to_pil_image(u(input[0]))
            # 	im.save('results/btest_snow_%s.png' % i)
            # 	del im

            if device_ids:
                input = input.cuda(device_ids[0])
            c_label = label.numpy()

            logits = models(input)
            # print(logits.shape, logits[:,:11].shape)
            # exit()
            # if dataset in ['robin','pseudorobin'] and logits.shape[1]==12:
            # predictions = logits[:,:11].argmax(dim=1)
            # else:
            predictions = logits.argmax(dim=1)
            # print(predictions.cpu().data, c_label, logits.cpu().numpy(), "\n")

            # Predictions are optional. If you do not specify them,
            # they will be computed within the loss function.
            loss = rpl_loss(logits, predictions)
            t_loss+=loss

            # When using self-learning, you need to add an additional optimizer
            # step in your evaluation loop.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            models_differ = 0

            #### checking which parameters change
            if check_changed_pmtr:
                if i == 200:
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
            #### 

            # output, *_ = models(input)
            out = predictions.cpu().numpy()
            # print(out, c_label, out.argmax())
            # scores = np.concatenate((scores,out))
            # out = out.argmax()
            if c_label.shape[0]>1:
                for cl in range(len(categories)):
                    correct[cl] += np.sum((out == c_label)*(c_label==cl))
                    total_samples[cl] += np.sum(c_label==cl)
            else:
                correct[c_label] += np.sum(out == c_label)
                total_samples[c_label] += 1
            # if np.sum(total_samples)>=300:
            #     break

        for i in range(nclasses):
            if total_samples[i]>0:
                print('Class {}: {:1.3f}'.format(categories_train[i],correct[i]/total_samples[i]))
        test_acc = (np.sum(correct)/np.sum(total_samples))
        print("Epoch {} Acc: {} Loss:{}".format(epoch, test_acc, t_loss/np.sum(total_samples)))
    
    if save_checkpt:
        best_check = {
                        'state_dict': models.state_dict(),
                        'test_acc': test_acc,
                        'epoch': epoch
                    }
        save_checkpoint(best_check, 'baseline_models/{}_adapted_{}_{}.pth'.format(corr,backbone_type, dataset), True)
    # exit()
    return test_acc, scores, models

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
    elif backbone_type=='resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, len(categories_train))
        # model.fc = torch.nn.Linear(model.fc.in_features, 12)
    elif backbone_type == 'resnet18' or backbone_type == 'resnext' or backbone_type=='densenet':
        raise NotImplementedError("baselines {} not implemented".format(backbone_type))
    else:
        raise(NotImplementedError)

    if device_ids:
        load_dict = torch.load(path, map_location='cuda:{}'.format(0))
    else:
        load_dict = torch.load(path, map_location='cpu')

    model.load_state_dict(load_dict['state_dict'])

    model.cuda().eval() #* IS THIS NEEDED?

    ############################
    # for occ_level in occ_levels:

    #     if occ_level == 'ZERO':
    #         occ_types = ['']
    #     else:
    #         if dataset=='pascal3d+':
    #             occ_types = ['']#['_white','_noise', '_texture', '']
    #         elif dataset=='coco':
    #             occ_types = ['']

    #     for index, occ_type in enumerate(occ_types):
    #         # load images
    #         if corr is not None:
    #             print("Loading corrupted data\n")
    #             test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
    #                 categories, occ_level, occ_type,bool_load_occ_mask=True, determinate=True, corruption=corr)
    #         else:
    #             print("Loading clean data\n")
    #             test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
    #                 categories, occ_level, occ_type,bool_load_occ_mask=True)
            
    #         #* removing an image due to errors
    #         if corr is not None:
    #             errs = ['data/pascal3d+_occ_snow/carLEVELONE/n03770679_14513_2.JPEG', 'data/pascal3d+_occ_snow/carLEVELFIVE/n03770679_14513_2.JPEG', 'data/pascal3d+_occ_snow/carLEVELNINE/n03770679_14513_2.JPEG']
    #             for es in errs:
    #                 if es in test_imgs:
    #                     idx_rm = test_imgs.index(es)
    #                     del test_imgs[idx_rm]
    #                     del test_labels[idx_rm]
    #                     del masks[idx_rm]
    #         #*

    #         print('Total imgs for test of occ_level {} and occ_type {} '.format(occ_level, occ_type) + str(len(test_imgs)))
    #         """test_imgs is list of image path and name strings"""
    #         # get image loader
    #         test_imgset = Imgset(test_imgs, masks, test_labels, imgLoader, bool_square_images=False)
    #         # test_imgset = train_imgset
    #         # compute test accuracy
    #         acc, scores = test(models=model, test_data=test_imgset, batch_size=1)
    #         out_str = 'Test Model Name: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc)
    #         print(out_str)
    ############################
    # Adapt Loop
    ############################
    for occ_level in occ_levels_train:

        if occ_level == 'ZERO':
            occ_types = ['']
        else:
            if dataset=='pascal3d+':
                occ_types = ['']#['_white','_noise', '_texture', '']
            elif dataset=='coco':
                occ_types = ['']

        for index, occ_type in enumerate(occ_types):
            for cat in robin_cats:
                # load images
                if corr is not None:
                    print("Loading corrupted data\n")
                    train_imgs, train_labels, train_masks = getImg('train', categories_train, dataset,data_path, \
                        categories, occ_level, occ_type,bool_load_occ_mask=True, determinate=True, corruption=corr)
                    # test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
                    #     categories, occ_level, occ_type,bool_load_occ_mask=True, determinate=True, corruption=corr)
                else:
                    print("Loading clean data\n")
                    if dataset == 'robin':
                        if robin_all:
                            cat = None # for all categories
                        else:
                            cat = [cat]
                        print("Loading {} subcategory\n".format(cat))
                        train_imgs, train_labels, train_masks = getImg('test', categories_train, dataset,data_path, \
                            categories, occ_level, occ_type,bool_load_occ_mask=True, subcat=cat)
                    else:
                        train_imgs, train_labels, train_masks = getImg('train', categories_train, dataset,data_path, \
                            categories, occ_level, occ_type,bool_load_occ_mask=True)
                    # test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
                    #     categories, occ_level, occ_type,bool_load_occ_mask=True)
                
                #* removing an image due to errors
                # if corr is not None:
                #     errs = ['data/pascal3d+_occ_snow/carLEVELONE/n03770679_14513_2.JPEG', 'data/pascal3d+_occ_snow/carLEVELFIVE/n03770679_14513_2.JPEG', 'data/pascal3d+_occ_snow/carLEVELNINE/n03770679_14513_2.JPEG']
                #     for es in errs:
                #         if es in test_imgs:
                #             idx_rm = test_imgs.index(es)
                #             del test_imgs[idx_rm]
                #             del test_labels[idx_rm]
                #             del masks[idx_rm]
                #*

                # print('Total imgs for test of occ_level {} and occ_type {} '.format(occ_level, occ_type) + str(len(test_imgs)))
                # """test_imgs is list of image path and name strings"""
                # get image loader
                train_imgset = Imgset(train_imgs,train_masks, train_labels, imgLoader,bool_square_images=True)
                # test_imgset = Imgset(test_imgs, masks, test_labels, imgLoader, bool_square_images=False)
                # compute test accuracy
                acc, scores, model = adapt(models=model, test_data=train_imgset, batch_size=64, robust_da=rbsta)
                out_str = 'Adapt Model Name: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc)
                print(out_str)
            
                # acc, scores = test(models=model, test_data=test_imgset, batch_size=1)
                # out_str = 'Test Model Name: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc)
                # print(out_str)

model.cuda().eval()
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
        if corr is not None:
            print("Loading corrupted data\n")
            test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
                categories, occ_level, occ_type,bool_load_occ_mask=True, determinate=True, corruption=corr)
        else:
            print("Loading clean data\n")
            test_imgs, test_labels, masks = getImg('test', categories_train, dataset,data_path, \
                categories, occ_level, occ_type,bool_load_occ_mask=True)
        
        #* removing an image due to errors
        if corr is not None:
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
        # test_imgset = train_imgset
        # compute test accuracy
        acc, scores = test(models=model, test_data=test_imgset, batch_size=1)
        out_str = 'Test Model Name: Occ_level:{}, Occ_type:{}, Acc:{}'.format(occ_level, occ_type, acc)
        print(out_str)
    
