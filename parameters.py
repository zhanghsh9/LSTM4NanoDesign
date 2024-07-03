from torch import nn

# Parameters

# PATH
RESULTS_PATH = './results'
DATA_PATH = './data'
FIGS_PATH = 'figs'
MODEL_PATH = 'models'

# Dataset
RODS = 2
BATCH_SIZE = 32
# See https://discuss.pytorch.org/t/w-cudaipctypes-cpp-22-producer-process-has-been-terminated-before-all-shared-cuda-tensors-released-see-note-sharing-cuda-tensors/124445/11
NUM_WORKERS = 16
SAMPLE_RATE = 1

# Model
# See https://datascience.stackexchange.com/questions/93768/dimensions-of-transformer-dmodel-and-depth
ATTENTION = 3
HIDDEN_UNITS = [1024, 1024, 1024, 1024]
# DROPOUT = 0.1
NUM_LAYERS = [1] * len(HIDDEN_UNITS)
# NUM_LSTMS = 3
NUM_HEADS = 1
ACTIVATE_FUNC = nn.ReLU()

# Train
LEARNING_RATE = 1e-3
EPOCHS = 800
VALID_FREQ = int(EPOCHS / 8) if EPOCHS < 200 else int(EPOCHS / 16) if EPOCHS < 500 else int(EPOCHS / 32)
# Learning rate drop rate
STEP_SIZE = 250
GAMMA = 0.75
