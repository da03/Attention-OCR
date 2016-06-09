__author__ = 'moonkey'

import os
import numpy as np
from PIL import Image
# from keras.preprocessing.sequence import pad_sequences
from collections import Counter
import cPickle
import random


class BucketData(object):
    def __init__(self):
        self.max_width = 0
        self.max_label_len = 0
        self.data_list = []
        self.label_list = []
        self.file_list = []

    def append(self, datum, label, filename):
        self.data_list.append(datum)
        self.label_list.append(label)
        self.file_list.append(filename)

        self.max_width = max(datum.shape[-1], self.max_width)
        self.max_label_len = max(len(label), self.max_label_len)

        return len(self.data_list)

    def flush_out(self, bucket_specs, valid_target_length=float('inf'),
                  go_shift=1):
        # print self.max_width, self.max_label_len
        res = dict(bucket_id=None,
                   data=None, zero_paddings=None, encoder_mask=None,
                   decoder_inputs=None, target_weights=None)

        def get_bucket_id():
            for idx in range(0, len(bucket_specs)):
                if bucket_specs[idx][0] >= self.max_width / 4 - 1 \
                        and bucket_specs[idx][1] >= self.max_label_len:
                    return idx
            return None

        res['bucket_id'] = get_bucket_id()
        if res['bucket_id'] is None:
            self.data_list, self.label_list = [], []
            self.max_width, self.max_label_len = 0, 0
            return None

        encoder_input_len, decoder_input_len = bucket_specs[res['bucket_id']]

        # ENCODER PART
        res['data'] = np.array(self.data_list)
        real_len = max(self.max_width / 4 - 1, 0)
        padd_len = encoder_input_len - real_len
        res['zero_paddings'] = np.zeros([len(self.data_list), padd_len, 512],
                                        dtype=np.float32)
        encoder_mask = np.concatenate(
            (np.ones([len(self.data_list), real_len], dtype=np.float32),
             np.zeros([len(self.data_list), padd_len], dtype=np.float32)),
            axis=1)
        res['encoder_mask'] = [a[:, np.newaxis] for a in encoder_mask.T]  # 32, (100, )
        res['real_len'] = self.max_width

        # DECODER PART
        target_weights = []
        for l_idx in range(len(self.label_list)):
            label_len = len(self.label_list[l_idx])
            if label_len <= decoder_input_len:
                self.label_list[l_idx] = np.concatenate((
                    self.label_list[l_idx],
                    np.zeros(decoder_input_len - label_len, dtype=np.int32)))
                one_mask_len = min(label_len - go_shift, valid_target_length)
                target_weights.append(np.concatenate((
                    np.ones(one_mask_len, dtype=np.float32),
                    np.zeros(decoder_input_len - one_mask_len,
                             dtype=np.float32))))
            else:
                raise NotImplementedError
                # self.label_list[l_idx] = \
                # self.label_list[l_idx][:decoder_input_len]
                # target_weights.append([1]*decoder_input_len)

        res['decoder_inputs'] = [a.astype(np.int32) for a in
                                 np.array(self.label_list).T]
        res['target_weights'] = [a.astype(np.float32) for a in
                                 np.array(target_weights).T]
        #print (res['decoder_inputs'][0])
        #assert False
        assert len(res['decoder_inputs']) == len(res['target_weights'])
        res['filenames'] = self.file_list

        self.data_list, self.label_list, self.file_list = [], [], []
        self.max_width, self.max_label_len = 0, 0

        return res

    def __len__(self):
        return len(self.data_list)

    def __iadd__(self, other):
        self.data_list += other.data_list
        self.label_list += other.label_list
        self.max_label_len = max(self.max_label_len, other.max_label_len)
        self.max_width = max(self.max_width, other.max_width)

    def __add__(self, other):
        res = BucketData()
        res.data_list = self.data_list + other.data_list
        res.label_list = self.label_list + other.label_list
        res.max_width = max(self.max_width, other.max_width)
        res.max_label_len = max((self.max_label_len, other.max_label_len))
        return res
