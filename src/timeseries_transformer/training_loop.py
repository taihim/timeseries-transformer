"""Training script for the timeseries transformer model."""
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.timeseries_transformer.constants import (
                                              EPOCHS, BATCH_SIZE, USE_K_FOLD,
                                              NUM_FOLDS, SHUFFLE, EMBED_SIZE,
                                              NUM_ATTN_HEADS, ENCODER_FF_DIM,
                                              DROPOUT, NUM_ENCODER_BLOCKS
                                            )
from src.timeseries_transformer.datasets import DatasetBuilder
from src.timeseries_transformer.timeseries_model import EncoderClassifier
from src.timeseries_transformer.utils import save_model

eval_iters = 200


def train_one_epoch(model, epoch_index, train_data, val_data):
    """Train the model for one epoch."""
    train_running_loss = 0.
    train_last_loss = 0.
    train_correct = 0
    iterations = 0

    for i, (inputs, labels) in enumerate(train_data):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        model["opt"].zero_grad()
        outputs = model["model"](inputs)
        loss = model["loss_fn"](outputs, labels.to(torch.long).reshape(-1))
        loss.backward()
        model["opt"].step()
        train_running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct_labels = labels.squeeze()

        train_correct += (predictions == correct_labels).int().sum() / len(labels) * 100
        iterations += 1
    train_last_loss = train_running_loss / len(train_data)
    train_acc = (train_correct / iterations)

    val_running_loss = 0.
    val_last_loss = 0.
    val_correct = 0

    iterations = 0
    model["model"].eval()
    for i, (inputs, labels) in enumerate(val_data):
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        outputs = model["model"](inputs)
        loss = model["loss_fn"](outputs, labels.to(torch.long).reshape(-1))
        val_running_loss += loss.item()

        predictions = torch.argmax(outputs, dim=1)
        correct_labels = labels.squeeze()

        val_correct += (predictions == correct_labels).int().sum() / len(labels) * 100
        iterations += 1
    val_last_loss = val_running_loss / len(val_data)
    val_acc = (val_correct / iterations)

    return train_last_loss, train_acc, val_last_loss, val_acc


def train_k_fold(train_dataset_builder: DatasetBuilder):
    """Train the model using k-fold cross-validation."""
    train_dataloaders, val_dataloaders = train_dataset_builder.create_dataloader(
        batch_size=BATCH_SIZE,
        shuffle=SHUFFLE,
        drop_last=True
    )

    models: list[dict[str, EncoderClassifier | CrossEntropyLoss | Adam]] = []
    losses: list[list] = []
    accuracies: list[list] = []

    for fold in range(len(train_dataloaders)):
        print('FOLD {}:'.format(fold + 1))

        train_dataloader = train_dataloaders[fold]
        val_dataloader = val_dataloaders[fold]

        model = EncoderClassifier(
                    inputs=train_dataset_builder.shape,
                    embed_size=EMBED_SIZE,
                    num_heads=NUM_ATTN_HEADS,
                    ff_dim=ENCODER_FF_DIM,
                    dropout=DROPOUT,
                    num_blocks=NUM_ENCODER_BLOCKS
                )
        model.cuda()
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=0.001)
        models.append({"model": model, "loss_fn": criterion, "opt": optimizer})

        losses.append([])
        accuracies.append([])

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch + 1))

            models[fold]["model"].train()
            train_avg_loss, train_acc, val_avg_loss, val_acc = train_one_epoch(
                                                                            models[fold],
                                                                            epoch,
                                                                            train_dataloader,
                                                                            val_dataloader,
                                                                        )

            print(f"Training loss: {train_avg_loss}")
            print(f"Training accuracy: {str(train_acc)}")
            print(f"Validation loss: {str(val_avg_loss)}")
            print(f"Validation accuracy: {val_acc}")

            losses[fold].append((train_avg_loss, val_avg_loss))
            accuracies[fold].append((train_acc, val_acc))

    save_model(model=models, path=)
    show_and_save_plots(losses, accuracies)
    save_metrics()
def train(train_db):
    """Train the model using a single train and test split."""
    train_dataloader = train_db.create_dataloader(batch_size=BATCH_SIZE, drop_last=True)



if __name__ == "__main__":
    if torch.cuda.is_available():
        cuda0 = torch.device('cuda:0')

    train_db = DatasetBuilder(split="train", use_k_fold=USE_K_FOLD, num_folds=NUM_FOLDS)
    test_db = DatasetBuilder(split="test")
    test_dataloader = test_db.create_dataloader(batch_size=BATCH_SIZE, drop_last=True)

    if USE_K_FOLD:
        train_k_fold(train_db)
    else:
        train(train_db)


