#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2 python -m domainbed.scripts.sweep delete_incomplete \
       --data_dir=/home/zhaoxin/data/DG/domainbed \
       --output_dir=trainout/PACS \
       --command_launcher multi_gpu \
       --algorithms ERM \
       --datasets PACS \
       --n_hparams 20 \
       --n_trials 3 \
       --single_test_envs
