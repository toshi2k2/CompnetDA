#/ Create psuedo dataset (for ROBIN data only) from list of images saved
import os, sys

p = os.path.abspath('.')
sys.path.insert(1, p)
from Initialization_Code.config_initialization import vc_num, dataset, categories, data_path, \
    cat_test, device_ids, Astride, Apad, Arf, vMF_kappa, layer, init_path, \
    dict_dir, sim_dir, extractor, da_sim_dir, da_dict_dir, robin_cats, \
        model_save_dir, da_init_path
import pickle
from shutil import copyfile, rmtree
import argparse

parser = argparse.ArgumentParser(description='Mixture Model Calculation')
# parser.add_argument('--da', type=bool, default=True, help='running DA or not')
parser.add_argument('--corr', type=str, default=None, help='types of corruptions in dataset')
# parser.add_argument('--saved_model', type=str, default='baseline_models/Robin-train-vgg_bn.pth', help='loading saved model')
parser.add_argument('--bbone', type=str, default='vgg_tr', help="Backbone type")
parser.add_argument('--robin_cat', type=int, default=None, help='None-all robin subcategories, else number')
# parser.add_argument('--vcs', type=int, default=0, help='size of VCs used')
parser.add_argument('--of', type=bool, default=False, help='overwrite folder')
# parser.add_argument('--sveimglst', type=bool, default=False, help='save pseudo images list')
# parser.add_argument('--load', type=bool, default=True, help='Load pretrained models')
# parser.add_argument('--squareim', type=bool, default=True, help='use square images')
parser.add_argument('--dataset', type=str, default='robin', help='None-dataset in config files-else \
    the one you choose')

args = parser.parse_args()

if args.dataset is not None:
    dataset = args.dataset

overwrite_folder = args.of#True

# dataset='robin'
assert(dataset in ['robin','pseudorobin', 'pascal3d+'])
if dataset == 'robin':
    categories.remove('bottle')
    cat_test = categories
    # cat = [robin_cats[4]]
    # cat = None
    if args.robin_cat is None:
        cat = args.robin_cat#None
    else:
        cat=[robin_cats[args.robin_cat]]

    savename = 'data/Robin/cls_pseudo_test_all' #/For all subcats combined
    # outfile = './image_list_robin_da_True.npz'
    # outfile = './robin_all_psuedo_img2.pickle'
    # outfile = './robin_psuedo_img.pickle'
    outfile = da_init_path+'/{}_psuedooccludedrobin_img.pickle'.format(args.bbone)
    # filez = np.load(outfile)
    # filez.files
    # img_pth = filez['arr_0']
    # img_label = filez['arr_1']
elif dataset in ['pascal3d+']:
    cat_test = categories
    savename = 'data/pseudo_pascal3d+_occ'+args.corr
    outfile = da_init_path+'/{}_psuedopascal3d+_img.pickle'.format(args.bbone)

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
