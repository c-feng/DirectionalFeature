from easydict import EasyDict as edict
import numpy as np

__C = edict()
cfg = __C


# ============== general training config =====================
__C.TRAIN = edict()

# __C.TRAIN.NET = "unet.U_Net"
__C.TRAIN.NET = "unet_df.U_NetDF"

__C.TRAIN.LR = 0.001
__C.TRAIN.LR_CLIP = 0.00001
__C.TRAIN.DECAY_STEP_LIST = [60, 100, 150, 180] 
__C.TRAIN.LR_DECAY = 0.5

__C.TRAIN.GRAD_NORM_CLIP = 1.0

__C.TRAIN.OPTIMIZER = 'adam'
__C.TRAIN.WEIGHT_DECAY = 0  # "L2 regularization coeff [default: 0.0]"
__C.TRAIN.MOMENTUM = 0.9

# =============== model config ========================
__C.MODEL = edict()

__C.MODEL.SELFEATURE = True
__C.MODEL.SHIFT_N = 1
__C.MODEL.AUXSEG = True

# ================= dataset config ==========================
__C.DATASET = edict()

__C.DATASET.NAME = "acdc"
__C.DATASET.MEAN = 63.19523533061758
__C.DATASET.STD = 70.74166957523165

__C.DATASET.NUM_CLASS = 4

__C.DATASET.DF_USED = True
__C.DATASET.DF_NORM = True
__C.DATASET.BOUNDARY = False

__C.DATASET.TRAIN_LIST = "libs/datasets/jsonLists/acdcList/train.json"
__C.DATASET.TEST_LIST = "libs/datasets/jsonLists/acdcList/test.json"

__C.DATASET.TEST_PERSON_LIST = "libs/datasets/personList/AcdcTestPersonCarname.json"
