import os
import math

from StringIO import StringIO
from PIL import Image

import numpy as np
import tensorflow as tf

from .bucketdata import BucketData


class DataGen(object):
    GO_ID = 1
    EOS_ID = 2
    IMAGE_HEIGHT = 32

    def __init__(self,
                 annotation_fn,
                 evaluate=False,
                 valid_target_len=float('inf'),
                 img_width_range=(12, 320),
                 word_len=30,
                 epochs=1000):
        """
        :param annotation_fn:
        :param lexicon_fn:
        :param valid_target_len:
        :param img_width_range: only needed for training set
        :param word_len:
        :param epochs:
        :return:
        """
        self.epochs = epochs
        self.valid_target_len = valid_target_len
        self.bucket_min_width, self.bucket_max_width = img_width_range

        if os.path.exists(annotation_fn):
            self.annotation_path = annotation_fn
        else:
            raise IOError("The .tfrecords file %s does not exist." % annotation_fn)

        if evaluate:
            self.bucket_specs = [(int(math.floor(64 / 4)), int(word_len + 2)),
                                 (int(math.floor(108 / 4)), int(word_len + 2)),
                                 (int(math.floor(140 / 4)), int(word_len + 2)),
                                 (int(math.floor(256 / 4)), int(word_len + 2)),
                                 (int(math.floor(img_width_range[1] / 4)), int(word_len + 2))]
        else:
            self.bucket_specs = [(int(64 / 4), 9 + 2),
                                 (int(108 / 4), 15 + 2),
                                 (int(140 / 4), 17 + 2),
                                 (int(256 / 4), 20 + 2),
                                 (int(math.ceil(img_width_range[1] / 4)), word_len + 2)]

        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

        dataset = tf.contrib.data.TFRecordDataset([self.annotation_path])
        dataset = dataset.map(self._parse_record)
        dataset = dataset.shuffle(buffer_size=10000)
        self.dataset = dataset.repeat(self.epochs)

    def clear(self):
        self.bucket_data = {i: BucketData()
                            for i in range(self.bucket_max_width + 1)}

    def gen(self, batch_size):
        valid_target_len = self.valid_target_len

        dataset = self.dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()

        images, labels = iterator.get_next()

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            while True:
                try:
                    raw_images, raw_labels = sess.run([images, labels])
                    for img, lex in zip(raw_images, raw_labels):
                        word = self.convert_lex(lex)
                        if valid_target_len < float('inf'):
                            word = word[:valid_target_len + 1]

                        img_data = Image.open(StringIO(img))
                        width, height = img_data.size
                        resized_width = math.floor(float(width) / height * self.IMAGE_HEIGHT)

                        b_idx = min(resized_width, self.bucket_max_width)
                        bucket_size = self.bucket_data[b_idx].append(img, resized_width, word, lex)
                        if bucket_size >= batch_size:
                            bucket = self.bucket_data[b_idx].flush_out(
                                self.bucket_specs,
                                valid_target_length=valid_target_len,
                                go_shift=1)
                            if bucket is not None:
                                yield bucket
                            else:
                                assert False, 'no valid bucket of width %d' % resized_width
                except tf.errors.OutOfRangeError:
                    break

        self.clear()

    def convert_lex(self, lex):
        assert lex and len(lex) < self.bucket_specs[-1][1]

        word = [self.GO_ID]
        for char in lex:
            assert 96 < ord(char) < 123 or 47 < ord(char) < 58
            word.append(
                ord(char) - 97 + 13 if ord(char) > 96 else ord(char) - 48 + 3)
        word.append(self.EOS_ID)
        word = np.array(word, dtype=np.int32)

        return word

    @staticmethod
    def _parse_record(example_proto):
        features = tf.parse_single_example(
            example_proto,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string),
            })
        return features["image"], features["label"]
