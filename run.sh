#!/usr/bin/env bash

# train on iam (handwritten)
python src/launcher.py --data-base-dir=/ --data-path=/home/sivankeret/wolf_dir/Dev2/Datasets/iam-words/images/tmp_images_lists/trainset.txt --model-dir=Workplace --log-path=Workplace/log.txt --steps-per-checkpoint=200 --phase=train

# train on Synth90k subset toy example
python src/launcher.py --data-base-dir=data/sample --data-path=data/sample/sample.txt --model-dir=Workplace/model --log-path=Workplace/model_log.txt --steps-per-checkpoint=200 --phase=train --no-load-model

# train with load model
python src/launcher.py --data-base-dir=data/sample --data-path=data/sample/sample.txt --model-dir=Workplace --log-path=Workplace/log.txt --phase=train --load-model

python src/train.py --phase=train --train-data-path=data/sample/sample.txt --val-data-path=data/sample/sample.txt --train-data-base-dir=data/sample --val-data-base-dir=data/sample --log-path=Workplace/log_test.txt --model-dir=Workplace


# test on same subset toy example
python src/launcher.py --phase=test --data-path=data/sample/sample.txt --data-base-dir=data/sample --log-path=Workplace/log_test.txt --load-model --model-dir=Workplace --output-dir=Workplace/results



python src/test.py --phase=test --data-path=data/sample/sample.txt --data-base-dir=data/sample --log-path=Workplace/log_test.txt --model-dir=Workplace --output-dir=Workplace/results


python src/launcher.py --phase=train --data-path=/media/data2/sivankeret/Datasets/mnt/ramdisk/max/90kDICT32px/annotation_train_words.txt --data-base-dir=/media/data2/sivankeret/Datasets/mnt/ramdisk/max/90kDICT32px --log-path=Workplace/log_before_refactor.txt --model-dir=Workplace
