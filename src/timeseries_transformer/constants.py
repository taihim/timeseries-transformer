"""Constants used in the timeseries_transformer module."""
from typing import Final
from pathlib import Path

# Model parameters
EMBED_SIZE: Final[int] = 256  # dimension of the embedding layer
NUM_ATTN_HEADS: Final[int] = 4  # number of attention heads
ENCODER_FF_DIM: Final[int] = 4  # dimension of the feedforward layer in the encoder
NUM_ENCODER_BLOCKS: Final[int] = 2  # number of encoder blocks
MLP_UNITS: Final[list[int]] = [128]  # the size of the feedforward layer used to make predictions
MLP_DROPOUT: Final[float] = 0.4  # dropout in the feedforward layer
DROPOUT: Final[float] = 0.25  # dropout in the encoder layer

# Training parameters
BATCH_SIZE: Final[int] = 64  # batch size
EPOCHS: Final[int] = 1  # number of epochs
LEARNING_RATE: Final[float] = 1e-3  # learning rate
SEED: Final[int] = 42  # seed for reproducibility
USE_K_FOLD: Final[bool] = False  # whether to use k-fold cross-validation
NUM_FOLDS: Final[int] = 2  # number of folds for cross-validation
SHUFFLE: Final[bool] = True  # whether to shuffle the dataset when training
TEST_DATA_SAMPLES: Final[float] = 1  # number of examples in the test set

# Post training parameters
BASE_PATH_RESULTS: Final[Path] = Path("../../evaluation_results")
