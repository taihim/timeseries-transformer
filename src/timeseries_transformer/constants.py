"""Constants used in the timeseries_transformer module."""
from typing import Final

EMBED_SIZE: Final[int] = 256  # size of the embeddings
NUM_HEADS: Final[int] = 4  # number of attention heads
FF_DIM: Final[int] = 4  # dimension of the feedforward layer in the encoder
NUM_TRANSFORMER_BLOCKS: Final[int] = 2  # number of encoder blocks
MLP_UNITS: Final[list[int]] = [128]  # the size of the feedforward layer used to make predictions
MLP_DROPOUT: Final[float] = 0.4  # dropout in the feedforward layer
DROPOUT: Final[float] = 0.25  # dropout in the encoder
NUM_FOLDS: Final[int] = 2  # number of folds for cross-validation
BATCH_SIZE: Final[int] = 64  # batch size
EPOCHS: Final[int] = 10  # number of epochs
LEARNING_RATE: Final[float] = 1e-3  # learning rate
SEED: Final[int] = 42  # seed for reproducibility
