import os.path as osp
import torch.nn as nn
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_checkpoint
from dassl.data.data_manager import DataManager
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.clip import tokenize
from tqdm import tqdm
from pathlib import Path
_tokenizer = _Tokenizer()
from utils.model_utils import *
from utils.templates import *
import open_clip
import numpy as np
from typing import Tuple, List, Union, Any
# from clip import clip

CUSTOM_TEMPLATES = {
    "OxfordPets": oxford_pets,
    "OxfordFlowers": oxford_flower,
    "FGVCAircraft": aircraft,
    "DescribableTextures": dtd,
    "EuroSAT": eurosat,
    "RESISC45": eurosat,
    "StanfordCars": cars,
    "Food101": food101,
    "SUN397": sun397,
    "places365": sun397,
    "Caltech101": caltech,
    "CIFAR10_local": cifar10,
    "CIFAR100_local": cifar100,
    "UCF101": ucf101,
    "kinetics400": kinetics400,
    "ImageNet": imagenet,
    "ImageNetSketch": imagenet,
    "ImageNetV2": imagenet,
    "ImageNetA": imagenet,
    "ImageNetR": imagenet,
    "ImageNetGaussian": imagenet,
    "ImageNetDefocus": imagenet,
    "CUBS200": CUBS200,
}

class CLIP_Zero_Shot_adapt(nn.Module):

    def __init__(self, model, classes, templates, device='cuda', dataset_name=None, log=None, txt_cls = None, cfg=None):
        super(CLIP_Zero_Shot_adapt, self).__init__()
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.device = device
        self.classes = classes
        self.model = model.to(device)
        self.log = log
        self.args = None
        self.txt_cls = txt_cls
        self.templates = templates
        self.classes = [x.replace('_', ' ') for x in self.classes]

        if 'quickgelu' in self.cfg.MODEL.BACKBONE.NAME: # for meta_clip
            self.tokenizer = open_clip

        elif cfg.MODEL.BACKBONE.NAME in clip._MODELS:
            self.tokenizer = clip

        else:
            raise ValueError(f'Backbone {self.cfg.MODEL.BACKBONE.NAME} not supported')

        if cfg.text_emb == 's_temp':
            self.templates = ['a photo of a {}.']
            self.text_embeddings_for_zero_shot = self.txt_features(self.classes, self.templates)

        elif cfg.text_emb == 'ds_temp':
            self.templates = self.templates
            self.text_embeddings_for_zero_shot = self.txt_features(self.classes, self.templates)

        elif cfg.text_emb == 'mpvr':
            self.text_embeddings_for_zero_shot = self.zeroshot_classifier_gpt(self.classes, 'mpvr') # use when saved with parent prompts


        elif cfg.text_emb in ['waffle','waffle+','waffle++' ,'dclip']:
            pass
        else:
            raise NotImplementedError

    def image_features(self, images):
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return image_features

    def txt_features(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]  # format with class
                texts = self.tokenize_text(texts)
                class_embeddings = self.model.encode_text(texts.cuda())  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights.squeeze()


    def tokenize_text(self, texts):

        if 'quickgelu' in self.cfg.MODEL.BACKBONE.NAME:
            texts = self.tokenizer.tokenize(texts=texts).cuda()  # tokenize

        elif self.cfg.MODEL.BACKBONE.NAME in clip._MODELS:
            texts = self.tokenizer.tokenize(texts=texts, truncate=True).cuda()  # tokenize

        return texts


    def get_embd_text(self, text):

        texts = self.tokenize_text(text)
        class_embeddings = self.model.encode_text(texts.cuda())  # embed with text encoder
        class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding /= class_embedding.norm()

        return class_embedding

    def zeroshot_classifier_gpt(self, classnames, mode = str):

        assert mode == 'mpvr'

        assert self.cfg.llm_type in ['gpt', 'mixtral']


        path_to_file = f'./descriptions/{self.cfg.llm_type}/{self.dataset_name}.json'

        print('Reading descriptions from ::: ', path_to_file)

        with open(path_to_file) as f:
            gpt3_prompts = json.load(f)

        if self.dataset_name == 'SUN397' or self.dataset_name == 'UCF101' or \
                self.dataset_name == 'StanfordCars_' or self.dataset_name == 'OxfordPets' or \
                self.dataset_name == 'Food101':

            classnames = gpt3_prompts.keys()

        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = []
                for t in gpt3_prompts[classname]:
                    texts.append(t)
                texts = self.tokenize_text(texts)
                class_embeddings = self.model.encode_text(texts.cuda())  # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    def forward(self, x1):
        with torch.no_grad():
            out = x1.float() @ self.text_embeddings_for_zero_shot.float()
        return out


@TRAINER_REGISTRY.register()
class clip_adapt(TrainerX):

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")


        clip_model = load_clip_to_cpu(cfg)



        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            clip_model.float()
        print("Building ZERO-SHOT-MODEL CLIP")
        self.model = CLIP_Zero_Shot_adapt(model=clip_model, classes=classnames,
                                          templates=CUSTOM_TEMPLATES[cfg.DATASET.NAME], dataset_name = cfg.DATASET.NAME, txt_cls = cfg.txt_cls, cfg=cfg)
        self.register_model("adapt", self.model)
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def build_data_loader(self):
        """Create essential data-related attributes.

        A re-implementation of this method must create the
        same attributes (self.dm is optional).
        """
        te_transform = te_transform_clip

        dm = DataManager(self.cfg, custom_tfm_test=te_transform, custom_tfm_train=tr_transforms)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u  # optional, can be None
        self.val_loader = dm.val_loader  # optional, can be None
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname  # dict {label: classname}

        self.dm = dm

    def parse_batch_train(self, batch):

        if isinstance(batch, list):
            input = batch["img"]
            input = torch.stack(input)  # two views from dataloader
            input = input.to(self.device)
        else:
            input = batch['img']
            input = input.to(self.device)

        label = batch["label"]
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
