#!/usr/bin/env bash

python src/launcher.py \
	--phase=test \
	--data-path=/media/data2/sivankeret/Datasets/mnt/ramdisk/max/90kDICT32px/annotation_train_words.txt \
	--data-base-dir=/media/data2/sivankeret/Datasets/mnt/ramdisk/max/90kDICT32px \
	--log-path=log_01_16_test.txt \
	--attn-num-hidden 256 \
	--batch-size 64 \
	--model-dir=model_01_16 \
	--load-model \
	--num-epoch=3 \
	--gpu-id=1 \
	--output-dir=model_01_16/synth90 \
	--use-gru \
    --target-embedding-size=10
