# Parameters

# PATH
RESULTS_PATH = './results'
DATA_PATH = './data'
FIGS_PATH = 'figs'
MODEL_PATH = 'models'

# Dataset
RODS = 2
MAX_SEQ_LEN = 50
NUM_WORDS = 1003
BATCH_SIZE = 32
# See https://discuss.pytorch.org/t/w-cudaipctypes-cpp-22-producer-process-has-been-terminated-before-all-shared-cuda-tensors-released-see-note-sharing-cuda-tensors/124445/11
NUM_WORKERS = 8
SAMPLE_RATE = 10
PAD_TOKEN = 0
START_TOKEN = 1001
END_TOKEN = 1002

# Model
# See https://datascience.stackexchange.com/questions/93768/dimensions-of-transformer-dmodel-and-depth
HEADS = 6
DIM_K = 32
DIM_MODEL = DIM_K * HEADS
DIM_FEEDFORWARD = DIM_MODEL * 5 if DIM_MODEL < 256 else DIM_MODEL * 4
DROPOUT = 0.1
NUM_LAYERS = 6

# Train
LEARNING_RATE = 5e-5
EPOCHS = 200
VALID_FREQ = int(EPOCHS / 8) if EPOCHS < 200 else int(EPOCHS / 16) if EPOCHS < 500 else int(EPOCHS / 32)
TF_RATIO = 0.5
