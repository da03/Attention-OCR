#!/usr/bin/env sh

python src/launcher.py --phase=train --data-path=/home/jeremy/data/IIIT5K-Word_V3.0/IIIT5K/train_file_label.txt --data-base-dir=/home/jeremy/data/IIIT5K-Word_V3.0/IIIT5K --log-path=log_0871.txt --attn-num-hidden 64 --batch-size 16 --model-dir=model_x --initial-learning-rate=1.0 --num-epoch=20000 --gpu-id=0
