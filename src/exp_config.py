import platform

"""
Default paramters for experiemnt
"""


class ExpConfig:
    
    GPU_ID = 0
    # phase 
    PHASE = 'test'
    VISUALIZE = True

    # input and output
    DATA_BASE_DIR = '/mnt/90kDICT32px'
    DATA_PATH = '/mnt/train_shuffled_words.txt' # path containing data file names and labels. Format: 
    MODEL_DIR = 'train' # the directory for saving and loading model parameters (structure is not stored)
    LOG_PATH = 'log.txt'
    OUTPUT_DIR = 'results' # output directory
    STEPS_PER_CHECKPOINT = 500 # checkpointing (print perplexity, save model) per how many steps

    # Optimization
    NUM_EPOCH = 1000
    BATCH_SIZE = 64
    INITIAL_LEARNING_RATE = 1.0 # initial learning rate, note the we use AdaDelta, so the initial value doe not matter much

    # Network parameters
    TARGET_EMBEDDING_SIZE = 10 # embedding dimension for each target
    ATTN_USE_LSTM = True # whether or not use LSTM attention decoder cell
    ATTN_NUM_HIDDEN=128 # number of hidden units in attention decoder cell
    ATTN_NUM_LAYERS = 2 # number of layers in attention decoder cell
                        # (Encoder number of hidden units will be ATTN_NUM_HIDDEN*ATTN_NUM_LAYERS)
    LOAD_MODEL = True
    OLD_MODEL_VERSION = False
    TARGET_VOCAB_SIZE = 26+10+3 # 0: PADDING, 1: GO, 2: EOS, >2: 0-9, a-z
