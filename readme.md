[//]: # (Official codebase for our paper: **Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs**.)

[//]: # (This repository contains the code for all the experiments &#40;for 20 datasets&#41; conducted in our paper.)

# Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs

This is the official repository for our paper [Meta Prompting](https://arxiv.org/pdf/2403.11755.pdf) which has been accepted 
for publication at ECCV 2024. 

In this paper, we present Meta-Prompting for Visual Recognition (MPVR), a method to effectively take humans out of the loop and completely automate the prompt generation process for zero-shot 
recognition. Taking as input only
minimal information about the target task, in the form of its short natural 
language description, and a list of associated class labels, MPVR
automatically produces a diverse set of category-specific prompts resulting 
in a strong zero-shot classifier. MPVR generalizes effectively across
various popular zero-shot image recognition benchmarks belonging to
widely different domains when tested with multiple LLMs and VLMs.
For example, MPVR obtains a zero-shot recognition improvement over
CLIP by up to 19.8% and 18.2% (5.0% and 4.5% on average over 20
datasets) leveraging GPT and Mixtral LLMs, respectively.
## Installation

Our code is built upon the official codebase of the [CoOp](https://github.dev/KaiyangZhou/CoOp).

As a first step, install `dassl` library (under `Meta-Prompting/`) in your environment by following the instructions [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation).

To further install all other dependencies, please run the following command, after having your environment activated:

```
pip install -r requirements.txt
```

## Datasets

Under `Meta-Prompting/` first make an empty data folder: 

```
mkdir data
```

Then download and structure your datasets according to the instructions provided in 
the [CoOp](https://github.dev/KaiyangZhou/CoOp)
official repository. 
Most of the datasets are already implemented in their codebase. 
For other datasets, you will need to download the datasets from the official sources and structure them as the other 
datasets in the `CoOp` codebase. For convenience, we provide the download links for remaining datasets here: 

1. [places365](http://places2.csail.mit.edu/download.html)
2. [cub200](https://www.vision.caltech.edu/datasets/cub_200_2011/)
3. [resisc45](https://meta-album.github.io/datasets/RESISC.html)
4. [k400](https://github.com/cvdfoundation/kinetics-dataset)
5. [oxfordpets](https://www.robots.ox.ac.uk/~vgg/data/pets/)

## 2.5M Category-Level VLM Prompts

Since the paper is currently under review at a conference, we only provide limited access to the VLM prompts used in our paper.
To gain access, please write an email with the subject **"MPVR VLM Prompts"** to M. Jehanzeb Mirza:

- Email: [jehanzeb95@gmail.com](mailto:jehanzeb95@gmail.com)



After the download, please `unzip` the files and place them in the `Meta-Prompting/descriptions` directory. This is required to run the MPVR experiments.
To generate the VLM prompts yourself for the datasets, please run the files for the individual dataset files present in the 
`Meta-Prompting/generate` directory.



## Experiments

In the following we provide instructions to obtain the baseline and MPVR results for the 20 datasets for all the models
used in our paper.


#### Baseline Results

1. To get the baseline results (with default ```a photo of a {}``` template) for 20 datasets, run the following command:

 
```  
bash scripts/zero_shot.sh s_temp none clip_b32 eurosat imagenet_r \
                           oxford_flowers imagenet_sketch dtd fgvc_aircraft food101 k400 caltech101 \
                           places365 cubs imagenet stanford_cars sun397 imagenetv2 cifar10 cifar100 \
                           oxford_pets ucf101 resisc

```

2. To get the baseline results (with dataset-specific templates) for 20 datasets, run the following command:

```  
bash scripts/zero_shot.sh ds_temp none clip_b32 eurosat imagenet_r \
                           oxford_flowers imagenet_sketch dtd fgvc_aircraft food101 k400 caltech101 \
                           places365 cubs imagenet stanford_cars sun397 imagenetv2 cifar10 cifar100 \
                           oxford_pets ucf101 resisc
```

#### MPVR Results

1. To get the MPVR results (with GPT prompts) for 20 datasets, run the following command:

```  
bash scripts/zero_shot.sh mpvr gpt clip_b32 eurosat imagenet_r \
                           oxford_flowers imagenet_sketch dtd fgvc_aircraft food101 k400 caltech101 \
                           places365 cubs imagenet stanford_cars sun397 imagenetv2 cifar10 cifar100 \
                           oxford_pets ucf101 resisc
```

2. To get the MPVR results (with Mixtral prompts) for 20 datasets, run the following command:

```  
bash scripts/zero_shot.sh mpvr mixtral clip_b32 eurosat imagenet_r \
                           oxford_flowers imagenet_sketch dtd fgvc_aircraft food101 k400 caltech101 \
                           places365 cubs imagenet stanford_cars sun397 imagenetv2 cifar10 cifar100 \
                           oxford_pets ucf101 resisc
```


In the above commands, change the model name with the desired models from the following list:

- `clip_b32` (OpenAI)
- `clip_b16` (OpenAI)
- `clip_l14` (OpenAI)
- `metaclip_b32` (MetaCLIP)
- `metaclip_b16` (MetaCLIP)
- `metaclip_l14` (MetaCLIP)

### To cite us: 
```bibtex
@inproceedings{mirza2024mpvr,
    author    = {Mirza, M. Jehanzeb and Karlinsky, Leonid and Lin, Wei and Doveh, Sivan and 
                 and Micorek, Jakub and Kozinski, Mateusz and Kuhene, Hilde and Possegger, Horst},
    booktitle = {Proceedings of the European Conference for Computer Vision (ECCV)},
    title     = {{Meta-Prompting for Automating Zero-shot Visual Recognition with LLMs}},
    year      = {2024}
    }
