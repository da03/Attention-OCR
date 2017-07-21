"""Visual Attention Based OCR Model."""

from __future__ import absolute_import
from __future__ import division

import time
import os
import math
import logging

import distance
import numpy as np
import tensorflow as tf

from PIL import Image
from six.moves import xrange  # pylint: disable=redefined-builtin

from .cnn import CNN
from .seq2seq_model import Seq2SeqModel
from ..util.data_gen import DataGen

tf.reset_default_graph()


class Model(object):
    SYMBOLS = '   0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def __init__(self,
                 phase,
                 visualize,
                 data_path,
                 output_dir,
                 batch_size,
                 initial_learning_rate,
                 num_epoch,
                 steps_per_checkpoint,
                 target_vocab_size,
                 model_dir,
                 target_embedding_size,
                 attn_num_hidden,
                 attn_num_layers,
                 clip_gradients,
                 max_gradient_norm,
                 session,
                 load_model,
                 gpu_id,
                 use_gru,
                 evaluate=False,
                 valid_target_length=float('inf'),
                 reg_val=0):

        gpu_device_id = '/gpu:' + str(gpu_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        logging.info('loading data')
        # load data
        if phase == 'train':
            self.s_gen = DataGen(
                data_path, valid_target_len=valid_target_length, evaluate=False,
                epochs=num_epoch)
        else:
            batch_size = 1
            self.s_gen = DataGen(data_path, evaluate=True, epochs=1)

        logging.info('phase: %s' % phase)
        logging.info('model_dir: %s' % (model_dir))
        logging.info('load_model: %s' % (load_model))
        logging.info('output_dir: %s' % (output_dir))
        logging.info('steps_per_checkpoint: %d' % (steps_per_checkpoint))
        logging.info('batch_size: %d' % (batch_size))
        logging.info('num_epoch: %d' % num_epoch)
        logging.info('learning_rate: %d' % initial_learning_rate)
        logging.info('reg_val: %d' % (reg_val))
        logging.info('max_gradient_norm: %f' % max_gradient_norm)
        logging.info('clip_gradients: %s' % clip_gradients)
        logging.info('valid_target_length %f' % valid_target_length)
        logging.info('target_vocab_size: %d' % target_vocab_size)
        logging.info('target_embedding_size: %f' % target_embedding_size)
        logging.info('attn_num_hidden: %d' % attn_num_hidden)
        logging.info('attn_num_layers: %d' % attn_num_layers)
        logging.info('visualize: %s' % visualize)

        buckets = self.s_gen.bucket_specs
        logging.info('buckets')
        logging.info(buckets)
        if use_gru:
            logging.info('using GRU in the decoder.')

        # variables

        self.zero_paddings = tf.placeholder(tf.float32, shape=(None, None, 512), name='zero_paddings')

        self.decoder_inputs = []
        self.encoder_masks = []
        self.target_weights = []
        for i in xrange(int(buckets[-1][0] + 1)):
            self.encoder_masks.append(tf.placeholder(tf.float32, shape=[None, 1],
                                      name="encoder_mask{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                       name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                       name="weight{0}".format(i)))


        self.reg_val = reg_val
        self.sess = session
        self.evaluate = evaluate
        self.steps_per_checkpoint = steps_per_checkpoint
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.buckets = buckets
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.global_step = tf.Variable(0, trainable=False)
        self.valid_target_length = valid_target_length
        self.phase = phase
        self.visualize = visualize
        self.learning_rate = initial_learning_rate
        self.clip_gradients = clip_gradients

        if phase == 'train':
            self.forward_only = False
        elif phase == 'test':
            self.forward_only = True
        else:
            assert False, phase


        # TODO: [32, 85] -- proportional resizing
        # TODO: one or many images

        # self.img_pl = tf.placeholder(tf.string, shape=None, name='input_image_as_bytes')
        # self.imgs_pl = tf.expand_dims(self.img_pl, 0, name='input_images_as_bytes')

        self.img_pl = tf.placeholder(tf.string, name='input_image_as_bytes')

        self.img_data = tf.cond(
            tf.less(tf.rank(self.img_pl), 1),
            lambda: tf.expand_dims(self.img_pl, 0),
            lambda: self.img_pl
        )

        self.img_data = tf.map_fn(lambda x: tf.image.decode_png(x, channels=1), self.img_data, dtype=tf.uint8)

        self.dims = tf.shape(self.img_data)
        height_const = tf.constant(DataGen.IMAGE_HEIGHT, dtype=tf.float32)
        new_height = tf.to_int32(height_const)
        new_width = tf.to_int32(tf.ceil(tf.to_float(self.dims[2]) / tf.to_float(self.dims[1]) * height_const))
        self.new_dims = [new_height, new_width]  # [32, 85]  #

        with tf.control_dependencies(self.new_dims), tf.device(gpu_device_id):
            self.img_data = tf.image.resize_images(self.img_data, self.new_dims, method=tf.image.ResizeMethod.BICUBIC)
            self.img_data = tf.transpose(self.img_data, perm=[0, 3, 1, 2])

        # with tf.device(gpu_device_id):
            cnn_model = CNN(self.img_data, True)
            self.conv_output = cnn_model.tf_output()
            self.concat_conv_output = tf.concat(axis=1, values=[self.conv_output, self.zero_paddings])
            self.perm_conv_output = tf.transpose(self.concat_conv_output, perm=[1, 0, 2])
            self.attention_decoder_model = Seq2SeqModel(
                encoder_masks = self.encoder_masks,
                encoder_inputs_tensor = self.perm_conv_output,
                decoder_inputs = self.decoder_inputs,
                target_weights = self.target_weights,
                target_vocab_size = target_vocab_size,
                buckets = buckets,
                target_embedding_size = target_embedding_size,
                attn_num_layers = attn_num_layers,
                attn_num_hidden = attn_num_hidden,
                forward_only = self.forward_only,
                use_gru = use_gru)

        if not self.forward_only:  # train
            self.updates = []
            self.summaries_by_bucket = []
            with tf.device(gpu_device_id):
                params = tf.trainable_variables()
                opt = tf.train.AdadeltaOptimizer(learning_rate=initial_learning_rate)
                for b in xrange(len(buckets)):
                    if self.reg_val > 0:
                        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                        logging.info('Adding %s regularization losses', len(reg_losses))
                        logging.debug('REGULARIZATION_LOSSES: %s', reg_losses)
                        loss_op = self.reg_val * tf.reduce_sum(reg_losses) + self.attention_decoder_model.losses[b]
                    else:
                        loss_op = self.attention_decoder_model.losses[b]

                    gradients, params = zip(*opt.compute_gradients(loss_op, params))
                    if self.clip_gradients:
                        gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    # Add summaries for loss, variables, gradients, gradient norms and total gradient norm.
                    summaries = []
                    summaries.append(tf.summary.scalar("loss", loss_op))
                    summaries.append(tf.summary.scalar("total_gradient_norm", tf.global_norm(gradients)))
                    all_summaries = tf.summary.merge(summaries)
                    self.summaries_by_bucket.append(all_summaries)
                    # update op - apply gradients
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_ops):
                        self.updates.append(opt.apply_gradients(zip(gradients, params), global_step=self.global_step))

        table = tf.contrib.lookup.MutableHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.string,
            default_value="",
            checkpoint=True,
        )

        insert = table.insert(
            tf.constant([i for i in xrange(len(self.SYMBOLS))], dtype=tf.int64),
            tf.constant(list(self.SYMBOLS)),
        )

        with tf.control_dependencies([insert]):

            output_num = []
            output_feed = []

            for b in xrange(len(buckets)):

                for l in xrange(len(self.attention_decoder_model.outputs[b])):
                    guess = tf.argmax(self.attention_decoder_model.outputs[b][l], axis=1)
                    output_num.append(guess)
                    output_feed.append(table.lookup(guess))

            tf.concat(output_num, 0)
            self.arr_prediction = tf.foldl(lambda a, x: a + x, output_feed)
            self.prediction = tf.gather(self.arr_prediction, 0, name='prediction')

        self.saver_all = tf.train.Saver(tf.all_variables())

        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and load_model:
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            self.saver_all.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            logging.info("Created model with fresh parameters.")
            self.sess.run(tf.initialize_all_variables())

    def test(self):
        step_time = 0.0
        loss = 0.0
        current_step = 1
        num_correct = 0
        num_total = 0

        for batch in self.s_gen.gen(self.batch_size):
            # Get a batch and make a step.
            start_time = time.time()
            result = self.step(batch, self.forward_only)
            loss += result['loss'] / self.steps_per_checkpoint
            grounds = [a for a in np.array([decoder_input.tolist() for decoder_input in batch['decoder_inputs']]).transpose()]
            step_outputs = [b for b in np.array([np.argmax(logit, axis=1).tolist() for logit in result['logits']]).transpose()]
            curr_step_time = (time.time() - start_time)

            logging.info('step_time: %f, loss: %f, step perplexity: %f'%(curr_step_time, result['loss'], math.exp(result['loss']) if result['loss'] < 300 else float('inf')))

            if self.visualize:
                step_attns = np.array([[a.tolist() for a in step_attn] for step_attn in result['attentions']]).transpose([1, 0, 2])

            for idx, output, ground in zip(range(len(grounds)), step_outputs, grounds):
                flag_ground, flag_out = True, True
                num_total += 1
                output_valid = []
                ground_valid = []
                for j in range(1, len(ground)):
                    s1 = output[j-1]
                    s2 = ground[j]
                    if s2 != 2 and flag_ground:
                        ground_valid.append(s2)
                    else:
                        flag_ground = False
                    if s1 != 2 and flag_out:
                        output_valid.append(s1)
                    else:
                        flag_out = False
                num_incorrect = distance.levenshtein(output_valid, ground_valid)
                num_incorrect = float(num_incorrect) / len(ground_valid)
                num_incorrect = min(1.0, num_incorrect)
                num_correct += 1. - num_incorrect

                if self.visualize:
                    self.visualize_attention(batch['file_list'][idx], step_attns[idx], output_valid, ground_valid, num_incorrect>0, batch['real_len'])

            precision = num_correct / self.batch_size
            logging.info('step %f - time: %f, loss: %f, perplexity: %f, precision: %f, batch_len: %f'
                         % (current_step, curr_step_time, result['loss'], math.exp(result['loss']) if result['loss'] < 300 else float('inf'), precision, batch['real_len']))
            current_step += 1

    def train(self):
        step_time = 0.0
        loss = 0.0
        current_step = 1
        writer = tf.summary.FileWriter(self.model_dir, self.sess.graph)

        logging.info('Starting the training process.')
        for batch in self.s_gen.gen(self.batch_size):
            start_time = time.time()
            result = self.step(batch, self.forward_only)
            loss += result['loss'] / self.steps_per_checkpoint
            grounds = [a for a in np.array([decoder_input.tolist() for decoder_input in batch['decoder_inputs']]).transpose()]
            step_outputs = [b for b in np.array([np.argmax(logit, axis=1).tolist() for logit in result['logits']]).transpose()]
            curr_step_time = (time.time() - start_time)
            step_time += curr_step_time / self.steps_per_checkpoint

            num_correct = 0
            for output, ground in zip(step_outputs, grounds):
                flag_ground, flag_out = True, True
                output_valid = []
                ground_valid = []
                for j in range(1, len(ground)):
                    s1 = output[j - 1]
                    s2 = ground[j]
                    if s2 != 2 and flag_ground:
                        ground_valid.append(s2)
                    else:
                        flag_ground = False
                    if s1 != 2 and flag_out:
                        output_valid.append(s1)
                    else:
                        flag_out = False
                num_incorrect = distance.levenshtein(output_valid, ground_valid)
                num_incorrect = float(num_incorrect) / len(ground_valid)
                num_incorrect = min(1.0, num_incorrect)
                num_correct += 1. - num_incorrect

            writer.add_summary(result['gradients'], current_step)

            precision = num_correct / self.batch_size
            logging.info('step %f - time: %f, loss: %f, perplexity: %f, precision: %f, batch_len: %f'
                         % (current_step, curr_step_time, result['loss'], math.exp(result['loss']) if result['loss'] < 300 else float('inf'), precision, batch['real_len']))

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % self.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                logging.info("global step %d step-time %.2f loss %f  perplexity "
                        "%.2f" % (self.global_step.eval(), step_time, loss, perplexity))
                # Save checkpoint and reset timer and loss.
                checkpoint_path = os.path.join(self.model_dir, "model.ckpt")
                logging.info("Saving model, current_step: %d"%current_step)
                self.saver_all.save(self.sess, checkpoint_path, global_step=self.global_step)
                step_time, loss = 0.0, 0.0

            current_step += 1

    def to_savedmodel(self):
        raise NotImplementedError

    def to_frozengraph(self):
        raise NotImplementedError

    # step, read one batch, generate gradients
    def step(self, batch, forward_only):
        bucket_id = batch['bucket_id']
        img_data = batch['data']
        zero_paddings = batch['zero_paddings']
        decoder_inputs = batch['decoder_inputs']
        target_weights = batch['target_weights']
        encoder_masks = batch['encoder_mask']
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                    " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                    " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        input_feed[self.img_pl.name] = img_data
        input_feed[self.zero_paddings.name] = zero_paddings
        for l in xrange(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
        for l in xrange(int(encoder_size)):
            try:
                input_feed[self.encoder_masks[l].name] = encoder_masks[l]
            except:
                pass

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

        # TODO: merging into one op

        # Output feed: depends on whether we do a backward step or not.
        output_feed = [self.attention_decoder_model.losses[bucket_id]]  # Loss for this batch.
        for l in xrange(decoder_size):  # Output logits.
            output_feed.append(self.attention_decoder_model.outputs[bucket_id][l])

        if not forward_only:  # train
            output_feed += [self.summaries_by_bucket[bucket_id],
                            self.updates[bucket_id]]
        elif self.visualize:  # test and visualize
            output_feed += self.attention_decoder_model.attention_weights_histories[bucket_id]

        outputs = self.sess.run(output_feed, input_feed)

        res = {
            'loss': outputs[0],
            'logits': outputs[1:(1+decoder_size)],
        }

        if not forward_only:
            res['gradients'] = outputs[2+decoder_size]
        elif self.visualize:
            res['attentions'] = outputs[(2+decoder_size):]

        return res

    def visualize_attention(self, filename, attentions, output_valid, ground_valid, flag_incorrect, real_len):
        if flag_incorrect:
            output_dir = os.path.join(self.output_dir, 'incorrect')
        else:
            output_dir = os.path.join(self.output_dir, 'correct')
        output_dir = os.path.join(output_dir, filename.replace('/', '_'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, 'word.txt'), 'w') as fword:
            fword.write(' '.join([chr(c-13+97) if c-13+97>96 else chr(c-3+48) for c in ground_valid])+'\n')
            fword.write(' '.join([chr(c-13+97) if c-13+97>96 else chr(c-3+48) for c in output_valid]))
            with open(filename, 'rb') as img_file:
                img = Image.open(img_file)
                w, h = img.size
                h = 32
                img = img.resize(
                        (real_len, h),
                        Image.ANTIALIAS)
                img_data = np.asarray(img, dtype=np.uint8)
                for idx in range(len(output_valid)):
                    output_filename = os.path.join(output_dir, 'image_%d.jpg'%(idx))
                    attention = attentions[idx][:(int(real_len/4)-1)]
                    attention_orig = np.zeros(real_len)
                    for i in range(real_len):
                        if 0 < i/4-1 and i/4-1 < len(attention):
                            attention_orig[i] = attention[int(i/4)-1]
                    attention_orig = np.convolve(attention_orig, [0.199547,0.200226,0.200454,0.200226,0.199547], mode='same')
                    attention_orig = np.maximum(attention_orig, 0.3)
                    attention_out = np.zeros((h, real_len))
                    for i in range(real_len):
                        attention_out[:,i] = attention_orig[i]
                    if len(img_data.shape) == 3:
                        attention_out = attention_out[:,:,np.newaxis]
                    img_out_data = img_data * attention_out
                    img_out = Image.fromarray(img_out_data.astype(np.uint8))
                    img_out.save(output_filename)
