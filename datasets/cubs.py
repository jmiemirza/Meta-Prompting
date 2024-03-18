import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD
from .oxford_pets import OxfordPets
# from ..build import DATASET_REGISTRY
# from ..base_dataset import Datum, DatasetBase
import random
import os.path as osp

from dassl.utils import listdir_nohidden
import os
import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
# from ..build import DATASET_REGISTRY
# from ..base_dataset import Datum, DatasetBase
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
import datasets.cifar_classes as cifar_classes


@DATASET_REGISTRY.register()
class CUBS200(DatasetBase):
    dataset_dir = "CUB_200_2011/CUB_200_2011"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))



        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")

        self.image_dir_paths = os.path.join(self.dataset_dir, "images.txt")
        # self.split_path = os.path.join(self.dataset_dir, "split_zhou_EuroSAT.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")

        self.split_path = os.path.join(self.dataset_dir, 'train_test_split.txt')

        self.class_names = os.path.join(self.dataset_dir, "classes.txt")
        self.img_cls_label = os.path.join(self.dataset_dir, "image_class_labels.txt")

        all_img_ind = list()
        with open(self.img_cls_label, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                all_img_ind.append(int(use_train))

        indices_to_use = list()
        with open(self.split_path, 'r') as in_file:
            for line in in_file:
                idx, use_train = line.strip('\n').split(' ', 2)
                if not bool(int(use_train)):
                    indices_to_use.append(int(idx))
        indices_to_use = [i - 1 for i in indices_to_use]

        all_img_ind_to_cls = [all_img_ind[i] for i in indices_to_use]

        labels = [i - 1 for i in all_img_ind_to_cls]


        filenames_to_use = set()
        with open(self.image_dir_paths, 'r') as in_file:
            for line in in_file:
                idx, fn = line.strip('\n').split(' ', 2)
                if int(idx)-1 in indices_to_use:
                    filenames_to_use.add(fn)

        all_label = list()
        for f in list(filenames_to_use):
            label = f.split('.')[0]
            all_label.append(int(label)-1)

        with open(self.class_names, 'r') as in_file:
            all_cls = list()
            for line in in_file:
                idx, cls = line.strip('\n').split(' ', 2)
                all_cls.append(cls)

        cleaned_cls = list()
        for c in all_cls:

            cls_ = c.split('.')[1]

            if '_' in cls_:
                cls_ = cls_.replace('_', ' ')

            cleaned_cls.append(cls_)


        all_corresponding_cls = [cleaned_cls[i] for i in all_label]

        item = self.read_data(all_label, list(filenames_to_use), all_corresponding_cls) # remember its only suppored for test!!!

        num_shots = cfg.DATASET.NUM_SHOTS


        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    item = data["train"]
            else:
                item = self.generate_fewshot_dataset(item, num_shots=num_shots)
                data = {"train": item}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        super().__init__(train_x=item, val=item, test=item)

    def read_data(self, labels, data, classes):

        items_x = []

        for label, path, class_name in zip(labels, data, classes):


            path = self.dataset_dir + '/images/' + path


            item = Datum(impath=path, label=label, classname=class_name)
            items_x.append(item)

        return items_x