"""Training script for the timeseries transformer model."""
# TODO: reduce code repetition and abstract the training loop into a class
# Get rid of training loop duplicated code
from uuid import uuid4

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from time import time
from src.timeseries_transformer.constants import (
    EPOCHS, BATCH_SIZE, USE_K_FOLD,
    NUM_FOLDS, SHUFFLE, EMBED_SIZE,
    NUM_ATTN_HEADS, ENCODER_FF_DIM,
    DROPOUT, NUM_ENCODER_BLOCKS, LEARNING_RATE
)
from src.timeseries_transformer.datasets import DatasetBuilder
from src.timeseries_transformer.timeseries_model import EncoderClassifier
from src.timeseries_transformer.utils import save_model_and_results, evaluate_model

eval_iters = 200


def train_one_epoch(model, epoch_index, train_data, val_data=None):
    """Train the model for one epoch."""
    train_running_loss = 0.
    train_last_loss = 0.
    train_correct = 0
    iterations = 0

    time_start = time()
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

    if val_data:
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
        time_end = time()
        time_taken = time_end - time_start

        return train_last_loss, train_acc, val_last_loss, val_acc, time_taken

    time_end = time()
    time_taken = time_end - time_start

    return train_last_loss, train_acc, None, None, time_taken


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
    epoch_times: list[list] = []

    for fold in range(len(train_dataloaders)):
        print('FOLD {}:'.format(fold + 1))

        train_dataloader = train_dataloaders[fold]
        val_dataloader = val_dataloaders[fold]

        model = EncoderClassifier(
                    input_shape=train_dataset_builder.shape,
                    embed_size=EMBED_SIZE,
                    num_heads=NUM_ATTN_HEADS,
                    ff_dim=ENCODER_FF_DIM,
                    dropout=DROPOUT,
                    num_blocks=NUM_ENCODER_BLOCKS
                )
        model.cuda()
        criterion = CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
        models.append({"id": str(uuid4())[0:8], "model": model, "loss_fn": criterion, "opt": optimizer})

        losses.append([])
        accuracies.append([])
        epoch_times.append([])
        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch + 1))

            models[fold]["model"].train()
            train_loss, train_acc, val_loss, val_acc, epoch_time_taken = train_one_epoch(
                                                                            models[fold],
                                                                            epoch,
                                                                            train_dataloader,
                                                                            val_dataloader,
                                                                        )

            print(f"Training loss: {train_loss}")
            print(f"Training accuracy: {train_acc}")
            print(f"Validation loss: {val_loss}")
            print(f"Validation accuracy: {val_acc}")
            print(f"Time Taken: {epoch_time_taken}")

            losses[fold].append((train_loss, val_loss))
            accuracies[fold].append((float(train_acc.cpu()), float(val_acc.cpu())))
            epoch_times[fold].append(epoch_time_taken)
    return models, losses, accuracies, epoch_times


def train(train_dataset_builder: DatasetBuilder):
    """Train the model using a single train and test split."""
    train_dataloader = train_db.create_dataloader(batch_size=BATCH_SIZE, drop_last=True)

    losses: list[tuple] = []
    accuracies: list[tuple] = []
    epoch_times: list[float] = []

    model = EncoderClassifier(
        input_shape=train_dataset_builder.shape,
        embed_size=EMBED_SIZE,
        num_heads=NUM_ATTN_HEADS,
        ff_dim=ENCODER_FF_DIM,
        dropout=DROPOUT,
        num_blocks=NUM_ENCODER_BLOCKS
    )
    model.cuda()
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    print(sum(p.numel() for p in model.parameters()))
    model_dict = {"id": str(uuid4())[0:8], "model": model, "loss_fn": criterion, "opt": optimizer}

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))

        model_dict["model"].train()
        train_loss, train_acc, _, _, epoch_time_taken = train_one_epoch(
                                                                        model_dict,
                                                                        epoch,
                                                                        train_dataloader
                                                                    )

        print(f"Training loss: {train_loss}")
        print(f"Training accuracy: {train_acc}")
        print(f"Time Taken: {epoch_time_taken}")

        losses.append(train_loss)
        accuracies.append(float(train_acc.cpu()))
        epoch_times.append(epoch_time_taken)

    return model_dict, losses, accuracies, epoch_times

#todo: save plots, confusion matrix, classification report, and total_training time. also record
# time taken for training/validation to converge
if __name__ == "__main__":
    """Train the timeseries transformer model."""
    if torch.cuda.is_available():
        cuda0 = torch.device('cuda:0')

    train_db = DatasetBuilder(split="train", use_k_fold=USE_K_FOLD, num_folds=NUM_FOLDS)
    test_db = DatasetBuilder(split="test")
    test_dataloader = test_db.create_dataloader(batch_size=BATCH_SIZE, drop_last=True)

    if USE_K_FOLD:
        models, losses, accuracies, epoch_times = train_k_fold(train_db)
        for fold, model in enumerate(models):
            eval_results = evaluate_model(model, test_dataloader)
            save_model_and_results(
                                model=model, losses=losses[fold], accuracies=accuracies[fold],
                                epoch_times=epoch_times[fold], eval_results=eval_results
                               )
    else:
        model, losses, accuracies, epoch_times = train(train_db)
        eval_results = evaluate_model(model, test_dataloader)
        save_model_and_results(
            model=model, losses=losses, accuracies=accuracies,
            epoch_times=epoch_times, eval_results=eval_results
        )


