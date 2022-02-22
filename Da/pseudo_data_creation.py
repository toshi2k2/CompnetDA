#/ Create psuedo dataset (for ROBIN data only) from list of images saved
import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, \
    cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer, init_path, \
    dict_dir, sim_dir, extractor, da_sim_dir, da_dict_dir, robin_cats
import pickle
from shutil import copyfile, rmtree

overwrite_folder = True

dataset='robin'
assert(dataset in ['robin','pseudorobin'])
if dataset == 'robin':
    categories.remove('bottle')
    cat_test = categories
    # cat = [robin_cats[0]]
    cat = None

savename = 'data/Robin/cls_pseudo_test_all' #/For all subcats combined
# outfile = './image_list_robin_da_True.npz'
# outfile = './robin_all_psuedo_img2.pickle'
# outfile = './robin_psuedo_img.pickle'
outfile = './vgg_bn_psuedorobincontext_img.pickle'
# filez = np.load(outfile)
# filez.files
# img_pth = filez['arr_0']
# img_label = filez['arr_1']
with open(outfile, 'rb') as fh:
    img_pth, img_label = pickle.load(fh)

if overwrite_folder:
    print("DELETING ALL PREVIOUS DATA")
    for files in os.listdir(savename):
        path = os.path.join(savename, files)
        try:
            rmtree(path)
        except OSError:
            os.remove(path)
# exit()

if os.path.exists(savename) and not overwrite_folder:
    raise(FileExistsError)
os.makedirs(savename, exist_ok=True)

for category in categories:
    path = os.path.join(savename, category)
    os.makedirs(path, exist_ok=True)

# for cats in robin_cats:
#     new_path = os.path.join(path, cats)
#     os.makedirs(new_path)

for ix, (source_path, psuedo_label) in enumerate(zip(img_pth, img_label)):
    # print(psuedo_label)
    dst = savename+'/'+categories[psuedo_label[0]]+'/'+str(ix)+'.jpg'
    copyfile(source_path, dst)
