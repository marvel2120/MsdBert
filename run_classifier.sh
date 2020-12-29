#!/usr/bin/env bash
for i in 'ResNetOnly' #'Res_BERT' 'MsdBERT' 'BertOnly'
do
    echo ${i}
    PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=1 python run_classifier.py --data_dir \
    ./data/ --image_dir ./images/ --output_dir ./output/${i}_output/  --do_train --do_test --model_select ${i} > ./log_file/${i}_log.txt 2>&1
done

