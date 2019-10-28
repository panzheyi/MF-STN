PARAM_PATH  = '../param/'
DATA_PATH   = '../data/'
VIZ_PATH = '../viz/'

TRAIN_PROP  = 0.8
EVAL_PROP   = 0.1

ROWS = 32
COLUMES = 32

TRAJ_LEN = 50
TRAJ_MOBILITY_BOUND = 3

FLOW_INPUT_LEN = 12
FLOW_INPUT_DIM = 3
FLOW_OUTPUT_LEN = 3
FLOW_OUTPUT_DIM = 2

FLOW_INPUT_PRE_DAYS = 4
FLOW_INPUT_PRE_DAYS_KERNEL = 3

from datetime import datetime
DATE_START = datetime(2015,2,1)
DATE_END = datetime(2015,7,1)