"""
Default parameters
"""


class Config:

    GPU_ID = 0
    VISUALIZE = False

    # I/O
    NEW_DATASET_PATH = 'dataset.tfrecords'
    DATA_PATH = 'data.tfrecords'
    MODEL_DIR = 'models'
    LOG_PATH = 'attentionocr.log'
    OUTPUT_DIR = 'results'
    STEPS_PER_CHECKPOINT = 500
    EXPORT_FORMAT = 'savedmodel'
    EXPORT_PATH = 'exported'

    # Optimization
    NUM_EPOCH = 1000
    BATCH_SIZE = 45
    INITIAL_LEARNING_RATE = 1.0

    # Network parameters
    CLIP_GRADIENTS = True  # whether to perform gradient clipping
    MAX_GRADIENT_NORM = 5.0  # Clip gradients to this norm
    TARGET_EMBEDDING_SIZE = 10  # embedding dimension for each target
    ATTN_USE_LSTM = True  # whether or not use LSTM attention decoder cell
    ATTN_NUM_HIDDEN = 128  # number of hidden units in attention decoder cell
    ATTN_NUM_LAYERS = 2  # number of layers in attention decoder cell
    # (Encoder number of hidden units will be ATTN_NUM_HIDDEN*ATTN_NUM_LAYERS)
    LOAD_MODEL = True
    OLD_MODEL_VERSION = False
    TARGET_VOCAB_SIZE = 26+10+3  # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z
