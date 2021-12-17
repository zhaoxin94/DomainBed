#!/bin/bash

gpu=$1

CUDA_VISIBLE_DEVICES=$gpu python -m domainbed.scripts.sweep launch \
       --data_dir=/home/zhaoxin/data/DG/domainbed \
       --output_dir=trainout/PACS_amp \
       --command_launcher local \
       --algorithms ERM FDA \
       --datasets PACS \
       --n_hparams 1 \
       --n_trials 3 \
       --single_test_envs