# Parameters

# PATH
RESULTS_PATH = './results'
DATA_PATH = './data'
FIGS_PATH = 'figs'
MODEL_PATH = 'models'

# Dataset
RODS = 2
BATCH_SIZE = 16
# See https://discuss.pytorch.org/t/w-cudaipctypes-cpp-22-producer-process-has-been-terminated-before-all-shared-cuda-tensors-released-see-note-sharing-cuda-tensors/124445/11
NUM_WORKERS = 8
SAMPLE_RATE = 10

# Model
# See https://datascience.stackexchange.com/questions/93768/dimensions-of-transformer-dmodel-and-depth
ATTENTION = 10
HIDDEN_UNITS = 1024
DROPOUT = 0.1
NUM_LAYERS = 1
NUM_LSTMS = 3

# Train
LEARNING_RATE = 1e-3
EPOCHS = 200
VALID_FREQ = int(EPOCHS / 8) if EPOCHS < 200 else int(EPOCHS / 16) if EPOCHS < 500 else int(EPOCHS / 32)
STEP_SIZE = 30
GAMMA = 0.8
