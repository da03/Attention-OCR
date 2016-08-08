#!/usr/bin/env sh

python src/launcher.py \
	--phase=train \
	--data-path=/home/jeremy/data/Synthetic_Word_Dataset/90kDICT32px/new_annotation_train.txt \
	--data-base-dir=/home/jeremy/data/Synthetic_Word_Dataset/90kDICT32px \
	--log-path=log_sy.txt \
	--attn-num-hidden 256 \
	--batch-size 64 \
	--model-dir=model_x \
	--initial-learning-rate=1.0 \
	--num-epoch=20000 \
	--gpu-id=0 \
	--use-gru \
        --target-embedding-size=10

