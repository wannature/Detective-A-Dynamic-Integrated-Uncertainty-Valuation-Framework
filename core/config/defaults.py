from yacs.config import CfgNode as CN

###########################
# Config definition
###########################

_C = CN()

_C.VERSION = 1



# Directory to save the output files
_C.OUTPUT_DIR = './output'
_C.LOG_NAME = 'log.txt'
# Path to a directory where the files were saved previously
_C.RESUME = ''
# Set seed to negative value to randomize everything
# Set seed to positive value to use a fixed seed
_C.SEED = -1
_C.GPU_ID = 4

###########################
# Input
###########################
_C.INPUT = CN()
_C.INPUT.SIZE = (256, 256)
_C.INPUT.CROP_SIZE = (224, 224)
# Mode of interpolation in resize functions
_C.INPUT.INTERPOLATION = 'bilinear'
# For available choices please refer to transforms.py
_C.INPUT.SOURCE_TRANSFORMS = ('random_crop', 'normalize')  # source training set
_C.INPUT.TARGET_TRANSFORMS = ('random_crop', 'normalize')  # target training set
_C.INPUT.TEST_TRANSFORMS = ('center_crop', 'normalize')  # target test set
# Default mean and std come from ImageNet
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
# Padding for random crop
_C.INPUT.CROP_PADDING = None
# ColorJitter (brightness, contrast, saturation, hue)
_C.INPUT.COLORJITTER_SCALAR = 0.5  # 0.1, 0.3, 0.5, 0.8, 1.0

###########################
# Dataset
###########################
_C.DATASET = CN()
# Directory where datasets are stored
_C.DATASET.ROOT = ''
_C.DATASET.NAME = ''
# List of domains
_C.DATASET.SOURCE_DOMAINS = []
_C.DATASET.TARGET_DOMAINS = []
_C.DATASET.SOURCE_TRAIN_DOMAIN = ''
_C.DATASET.TARGET_TRAIN_DOMAIN = ''
_C.DATASET.TARGET_VAL_DOMAIN = ''

# Number of class
_C.DATASET.NUM_CLASS = 12

###########################
# Dataloader
###########################
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
# _C.DATALOADER.NUM_WORKERS = 4
# Setting for the source data-loader
_C.DATALOADER.SOURCE = CN()
_C.DATALOADER.SOURCE.BATCH_SIZE = 32
# Setting for the target data-loader
_C.DATALOADER.TARGET = CN()
_C.DATALOADER.TARGET.BATCH_SIZE = 32
# Setting for the test data-loader
_C.DATALOADER.TEST = CN()
_C.DATALOADER.TEST.BATCH_SIZE = 32

###########################
# Model
###########################
_C.MODEL = CN()
# Path to model weights (for initialization)
_C.MODEL.INIT_WEIGHTS = ''
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet50'
_C.MODEL.BACKBONE.PRETRAINED = True

###########################
# Optimization
###########################
_C.OPTIM = CN()
_C.OPTIM.NAME = 'SGD'
_C.OPTIM.LR = 0.1
_C.OPTIM.GAMMA = 0.001
_C.OPTIM.DECAY_RATE = 0.75

# ###########################
# # Train
# ###########################
_C.TRAIN = CN()
# # How often (batch) to print training information
_C.TRAIN.PRINT_FREQ = 50
_C.TRAIN.TEST_FREQ = 5

###########################
# Trainer specifics
###########################
_C.TRAINER = CN()
_C.TRAINER.NAME = 'Detective'
_C.TRAINER.MAX_EPOCHS = 20
_C.TRAINER.BETA = 1.0
_C.TRAINER.LAMBDA = 0.05
# _C.TRAINER.ACTIVE_ROUND = [1, 2, 3, 4, 5]
_C.TRAINER.ACTIVE_ROUND = [10, 12, 14, 16, 18]
_C.TRAINER.KAPPA = 2
_C.TRAINER.BUDGET = 0.01
_C.TRAINER.CLIP_GRAD_NORM = -1.0

_C.NETWORK = CN()
_C.NETWORK.Z_DIM = 256
_C.NETWORK.FROZEN = False

_C.UNCERTAINTY = CN()
_C.UNCERTAINTY.LAMBDA_1 = 7
_C.UNCERTAINTY.LAMBDA_2 = 0.5

_C.SAVE = True
