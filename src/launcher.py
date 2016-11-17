__author__ = 'moonkey'

import sys, argparse, logging

import numpy as np
from PIL import Image
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
import keras.backend as K
K.set_session(sess)

from model.model import Model
import exp_config

def process_args(args, defaults):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu-id', dest="gpu_id",
                        type=int, default=defaults.GPU_ID)

    parser.add_argument('--use-gru', dest='use_gru', action='store_true')

    parser.add_argument('--phase', dest="phase",
                        type=str, default=defaults.PHASE,
                        choices=['train', 'test'],
                        help=('Phase of experiment, can be either' 
                            ' train or test, default=%s'%(defaults.PHASE)))
    parser.add_argument('--data-path', dest="data_path",
                        type=str, default=defaults.DATA_PATH,
                        help=('Path of file containing the path and labels'
                            ' of training or testing data, default=%s'
                            %(defaults.DATA_PATH)))
    parser.add_argument('--data-base-dir', dest="data_base_dir",
                        type=str, default=defaults.DATA_BASE_DIR,
                        help=('The base directory of the paths in the file '
                            'containing the path and labels, default=%s'
                            %(defaults.DATA_PATH)))
    parser.add_argument('--visualize', dest='visualize', action='store_true',
                        help=('Visualize attentions or not'
                            ', default=%s' %(defaults.VISUALIZE)))
    parser.add_argument('--no-visualize', dest='visualize', action='store_false')
    parser.set_defaults(visualize=defaults.VISUALIZE)
    parser.add_argument('--batch-size', dest="batch_size",
                        type=int, default=defaults.BATCH_SIZE,
                        help=('Batch size, default = %s'
                            %(defaults.BATCH_SIZE)))
    parser.add_argument('--initial-learning-rate', dest="initial_learning_rate",
                        type=float, default=defaults.INITIAL_LEARNING_RATE,
                        help=('Initial learning rate, default = %s'
                            %(defaults.INITIAL_LEARNING_RATE)))
    parser.add_argument('--num-epoch', dest="num_epoch",
                        type=int, default=defaults.NUM_EPOCH,
                        help=('Number of epochs, default = %s'
                            %(defaults.NUM_EPOCH)))
    parser.add_argument('--steps-per-checkpoint', dest="steps_per_checkpoint",
                        type=int, default=defaults.STEPS_PER_CHECKPOINT,
                        help=('Checkpointing (print perplexity, save model) per'
                            ' how many steps, default = %s'
                            %(defaults.STEPS_PER_CHECKPOINT)))
    parser.add_argument('--target-vocab-size', dest="target_vocab_size",
                        type=int, default=defaults.TARGET_VOCAB_SIZE,
                        help=('Target vocabulary size, default=%s' 
                            %(defaults.TARGET_VOCAB_SIZE)))
    parser.add_argument('--model-dir', dest="model_dir",
                        type=str, default=defaults.MODEL_DIR,
                        help=('The directory for saving and loading model '
                            '(structure is not stored), '
                            'default=%s' %(defaults.MODEL_DIR)))
    parser.add_argument('--target-embedding-size', dest="target_embedding_size",
                        type=int, default=defaults.TARGET_EMBEDDING_SIZE,
                        help=('Embedding dimension for each target, default=%s' 
                            %(defaults.TARGET_EMBEDDING_SIZE)))
    parser.add_argument('--attn-num-hidden', dest="attn_num_hidden",
                        type=int, default=defaults.ATTN_NUM_HIDDEN,
                        help=('number of hidden units in attention decoder cell'
                            ', default=%s' 
                            %(defaults.ATTN_NUM_HIDDEN)))
    parser.add_argument('--attn-num-layers', dest="attn_num_layers",
                        type=int, default=defaults.ATTN_NUM_LAYERS,
                        help=('number of hidden layers in attention decoder cell'
                            ', default=%s' 
                            %(defaults.ATTN_NUM_LAYERS)))
    parser.add_argument('--load-model', dest='load_model', action='store_true',
                        help=('Load model from model-dir or not'
                            ', default=%s' %(defaults.LOAD_MODEL)))
    parser.add_argument('--no-load-model', dest='load_model', action='store_false')
    parser.set_defaults(load_model=defaults.LOAD_MODEL)
    parser.add_argument('--old-model-version', dest='old_model_version', action='store_true',
                        help=('Whether the model was created by old keras version or not. Note that we need to make conversions for such old models.'
                            ', default=%s' %(defaults.OLD_MODEL_VERSION)))
    parser.add_argument('--no-old-model-version', dest='old_model_version', action='store_false')
    parser.set_defaults(old_model_version=defaults.OLD_MODEL_VERSION)
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default=defaults.LOG_PATH,
                        help=('Log file path, default=%s' 
                            %(defaults.LOG_PATH)))
    parser.add_argument('--output-dir', dest="output_dir",
                        type=str, default=defaults.OUTPUT_DIR,
                        help=('Output directory, default=%s' 
                            %(defaults.OUTPUT_DIR)))
    parameters = parser.parse_args(args)
    return parameters

def main(args, defaults):
    parameters = process_args(args, defaults)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    with sess.as_default():
        model = Model(
                phase = parameters.phase,
                visualize = parameters.visualize,
                data_path = parameters.data_path,
                data_base_dir = parameters.data_base_dir,
                output_dir = parameters.output_dir,
                batch_size = parameters.batch_size,
                initial_learning_rate = parameters.initial_learning_rate,
                num_epoch = parameters.num_epoch,
                steps_per_checkpoint = parameters.steps_per_checkpoint,
                target_vocab_size = parameters.target_vocab_size, 
                model_dir = parameters.model_dir,
                target_embedding_size = parameters.target_embedding_size,
                attn_num_hidden = parameters.attn_num_hidden,
                attn_num_layers = parameters.attn_num_layers,
                load_model = parameters.load_model,
                valid_target_length = float('inf'),
                gpu_id=parameters.gpu_id,
                use_gru=parameters.use_gru,
                session = sess,
                old_model_version = parameters.old_model_version)
        model.launch()

if __name__ == "__main__":
    main(sys.argv[1:], exp_config.ExpConfig)

