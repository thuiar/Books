## Introduction
We provide code of DTGNS. Please unzip data.zip and place it in this directory. The data paths are configured in config.py.

* requirement:

python                             3.8.3
numpy                              1.19.5
scikit-learn                       0.23.1
torch                              1.7.0
tqdm                               4.47.0
transformers                       3.4.0

### Parameters
dataset: semeval | fewrel
known_cls_ratio: 0.25 | 0.5 | 0.75 (default)  
labeled_ratio: 0.2 | 0.4 | 0.6 | 0.8 | 1.0 (default)  
seed: random seed (type: int)
#### An Example
python train.py --dataset semeval --known_cls_ratio 0.25 --labeled_ratio 1.0 --seed 0

#### Contact
If you have any questions, please contact zhaok7878@gmail.com