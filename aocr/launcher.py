import sys
import argparse
import logging

import tensorflow as tf

from .model.model import Model
from .defaults import Config
from .util import dataset

tf.logging.set_verbosity(tf.logging.ERROR)


def process_args(args, defaults):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Subcommands.')

    # Global arguments
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default=defaults.LOG_PATH,
                        help=('Log file path, default=%s'
                              % (defaults.LOG_PATH)))
    parser.set_defaults(visualize=defaults.VISUALIZE)
    parser.set_defaults(load_model=defaults.LOAD_MODEL)

    # Dataset generation
    parser_dataset = subparsers.add_parser('dataset', help='Create a dataset in the TFRecords format.')
    parser_dataset.set_defaults(phase='dataset')
    parser_dataset.add_argument('annotations_path', metavar='annotations',
                              type=str,
                              help=('Path to the annotation file'))
    parser_dataset.add_argument('output_path', nargs='?', metavar='output',
                              type=str, default=defaults.NEW_DATASET_PATH,
                              help=('Output path'
                                    ', default=%s'
                                    % (defaults.NEW_DATASET_PATH)))

    # Training
    parser_train = subparsers.add_parser('train', help='Train the model and save checkpoints.')
    parser_train.set_defaults(phase='train')
    parser_train.add_argument('dataset_path', metavar='dataset',
                              type=str, default=defaults.DATA_PATH,
                              help=('Training dataset in the TFRecords format'
                                    ', default=%s'
                                    % (defaults.DATA_PATH)))
    parser_train.add_argument('--no-resume', dest='load_model', action='store_false',
                              help=('Create an empty model even if checkpoints already exist.'
                                    ', default=%s' % (defaults.LOAD_MODEL)))

    # Testing
    parser_test = subparsers.add_parser('test', help='Test the saved model.')
    parser_test.set_defaults(phase='test')
    parser_test.add_argument('dataset_path', metavar='dataset',
                        type=str, default=defaults.DATA_PATH,
                        help=('Testing dataset in the TFRecords format'
                              ', default=%s'
                              % (defaults.DATA_PATH)))
    parser_test.add_argument('--visualize', dest='visualize', action='store_true',
                             help=('Visualize attentions'
                                   ', default=%s' % (defaults.VISUALIZE)))

    # Exporting
    parser_export = subparsers.add_parser('export', help='Export the saved checkpoints for production.')
    parser_export.add_argument('export_path', metavar='path',
                        type=str, default=defaults.EXPORT_PATH,
                        help=('Path to export the model in the specified format,'
                              'default=%s'
                              % (defaults.EXPORT_PATH)))
    parser_export.add_argument('--format', dest="format",
                               type=str, default=defaults.EXPORT_FORMAT,
                               choices=['frozengraph', 'savedmodel'],
                               help=('Export format for the model: either'
                                     'a frozen GraphDef or a SavedModel'
                                     '(default=%s)'
                                     % (defaults.EXPORT_FORMAT)))




    parser.add_argument('--gpu-id', dest="gpu_id",
                        type=int, default=defaults.GPU_ID)

    parser.add_argument('--use-gru', dest='use_gru', action='store_true')

    parser.add_argument('--batch-size', dest="batch_size",
                        type=int, default=defaults.BATCH_SIZE,
                        help=('Batch size, default = %s'
                              % (defaults.BATCH_SIZE)))
    parser.add_argument('--initial-learning-rate', dest="initial_learning_rate",
                        type=float, default=defaults.INITIAL_LEARNING_RATE,
                        help=('Initial learning rate, default = %s'
                              % (defaults.INITIAL_LEARNING_RATE)))
    parser.add_argument('--num-epoch', dest="num_epoch",
                        type=int, default=defaults.NUM_EPOCH,
                        help=('Number of epochs, default = %s'
                              % (defaults.NUM_EPOCH)))
    parser.add_argument('--steps-per-checkpoint', dest="steps_per_checkpoint",
                        type=int, default=defaults.STEPS_PER_CHECKPOINT,
                        help=('Checkpointing (print perplexity, save model) per'
                              ' how many steps, default = %s'
                              % (defaults.STEPS_PER_CHECKPOINT)))
    parser.add_argument('--target-vocab-size', dest="target_vocab_size",
                        type=int, default=defaults.TARGET_VOCAB_SIZE,
                        help=('Target vocabulary size, default=%s'
                              % (defaults.TARGET_VOCAB_SIZE)))
    parser.add_argument('--model-dir', dest="model_dir",
                        type=str, default=defaults.MODEL_DIR,
                        help=('The directory for saving and loading model '
                              'default=%s' %(defaults.MODEL_DIR)))
    parser.add_argument('--target-embedding-size', dest="target_embedding_size",
                        type=int, default=defaults.TARGET_EMBEDDING_SIZE,
                        help=('Embedding dimension for each target, default=%s'
                              % (defaults.TARGET_EMBEDDING_SIZE)))
    parser.add_argument('--attn-num-hidden', dest="attn_num_hidden",
                        type=int, default=defaults.ATTN_NUM_HIDDEN,
                        help=('number of hidden units in attention decoder cell'
                              ', default=%s'
                              % (defaults.ATTN_NUM_HIDDEN)))
    parser.add_argument('--attn-num-layers', dest="attn_num_layers",
                        type=int, default=defaults.ATTN_NUM_LAYERS,
                        help=('number of hidden layers in attention decoder cell'
                              ', default=%s'
                              % (defaults.ATTN_NUM_LAYERS)))
    parser.add_argument('--output-dir', dest="output_dir",
                        type=str, default=defaults.OUTPUT_DIR,
                        help=('Output directory, default=%s'
                              % (defaults.OUTPUT_DIR)))
    parser.add_argument('--max_gradient_norm', dest="max_gradient_norm",
                        type=int, default=defaults.MAX_GRADIENT_NORM,
                        help=('Clip gradients to this norm.'
                              ', default=%s'
                              % (defaults.MAX_GRADIENT_NORM)))
    parser.add_argument('--no-gradient_clipping', dest='clip_gradients', action='store_false',
                        help=('Do not perform gradient clipping, default for clip_gradients is %s' %
                              (defaults.CLIP_GRADIENTS)))
    parser.set_defaults(clip_gradients=defaults.CLIP_GRADIENTS)

    parameters = parser.parse_args(args)
    return parameters


def main(args):
    parameters = process_args(args, Config)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        if parameters.phase == 'dataset':
            dataset.generate(parameters.annotations_path, parameters.output_path)
            return

        model = Model(
            phase=parameters.phase,
            visualize=parameters.visualize,
            data_path=parameters.dataset_path,
            output_dir=parameters.output_dir,
            batch_size=parameters.batch_size,
            initial_learning_rate=parameters.initial_learning_rate,
            num_epoch=parameters.num_epoch,
            steps_per_checkpoint=parameters.steps_per_checkpoint,
            target_vocab_size=parameters.target_vocab_size,
            model_dir=parameters.model_dir,
            target_embedding_size=parameters.target_embedding_size,
            attn_num_hidden=parameters.attn_num_hidden,
            attn_num_layers=parameters.attn_num_layers,
            clip_gradients=parameters.clip_gradients,
            max_gradient_norm=parameters.max_gradient_norm,
            session=sess,
            load_model=parameters.load_model,
            gpu_id=parameters.gpu_id,
            use_gru=parameters.use_gru,
            )
        if parameters.phase == 'train':
            model.train()
        elif parameters.phase == 'test':
            model.test()
        else:
            raise NotImplementedError


if __name__ == "__main__":
    main(sys.argv[1:])
