#!/bin/bash
#$ -l rt_G.small=1
#$ -l h_rt=02:00:00
#$-j y
#$-m b
#$-m a
#$-m e
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/9.3.0 python/3.8 cuda/11.2 cudnn/8.1

export LD_LIBRARY_PATH=/lib:/usr/lib:/usr/local/lib:/apps/centos7/python/3.8.7/lib

python3 train_mt5.py\
    --project='regioSS'\
    --max_epochs=30\
    --model_name_or_path='google/mt5-small'\
    --tokenizer_name_or_path='kkuramitsu/mt5-pytoken'\
    --num_workers=4\
    --batch_size=16\
    data/codegen5_train.tsv data/codegen5_test.tsv data/talk_train.tsv

