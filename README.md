# MsdBert
This repository is an implementation for the paper "Modeling Intra and Inter-modality Incongruity for Multi-Modal Sarcasm Detection" that is published at the Findings of EMNLP-2020.

# How to use?
## Envs
```
torchvision==0.7.0+cu101
numpy==1.18.5
transformers==3.2.0
tqdm==4.49.0
wordninja==2.0.0
torch==1.6.0+cu101
Pillow==8.0.1
scikit_learn==0.24.0
```

## Install
```
pip3 install -r requirements.txt
```

## Dataset
You can find the Image data from https://github.com/headacheboy/data-of-multimodal-sarcasm-detection.

Put the images under a folder named "Images"

## Train and Test
Train:
```
python run_classifier.py --data_dir ./data/ --image_dir ./images/ --output_dir ./output/MsdBERT_output/  --do_train --do_test --model_select MsdBERT
```

Test:
```
python run_classifier.py --data_dir ./data/ --image_dir ./images/ --output_dir ./output/MsdBERT_output/  --do_test --model_select MsdBERT
```

Run all models recursively:
```
sh run_classifier.sh
```

## Saving
All the models and evaluation results will be saved under the "output" folder.

# Citation

``` @inproceedings{DBLP:conf/emnlp/PanL0Q020,
  author    = {Hongliang Pan and
               Zheng Lin and
               Peng Fu and
               Yatao Qi and
               Weiping Wang},
  title     = {Modeling Intra and Inter-modality Incongruity for Multi-Modal Sarcasm
               Detection},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural
               Language Processing: Findings, {EMNLP} 2020, Online Event, 16-20 November
               2020},
  pages     = {1383--1392},
  publisher = {Association for Computational Linguistics}
}
```




