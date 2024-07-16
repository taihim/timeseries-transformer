"""Training script for the timeseries transformer model."""
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.timeseries_transformer.constants import EPOCHS

eval_iters = 200

cuda0 = torch.device('cuda:0')

def train_one_epoch(epoch_index, train_data, val_data, model):
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

        predictions = torch.argmax(outputs, axis=1)
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

        predictions = torch.argmax(outputs, axis=1)
        correct_labels = labels.squeeze()

        val_correct += (predictions == correct_labels).int().sum() / len(labels) * 100
        iterations += 1
    val_last_loss = val_running_loss / len(val_data)
    val_acc = (val_correct / iterations)

    return train_last_loss, train_acc, val_last_loss, val_acc


epochs = 5
models = []
losses = []
accuracies = []
for fold in range(len(train_dataloaders)):
    print('FOLD {}:'.format(fold + 1))

    train_dataloader = train_dataloaders[fold]
    val_dataloader = val_dataloaders[fold]

    model = EncoderClassifier(inputs=xp, embed_size=embed_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout,
                              num_blocks=num_transformer_blocks)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    models.append({"model": model, "loss_fn": criterion, "opt": optimizer})

    losses.append([])
    accuracies.append([])

    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch + 1))

        models[fold]["model"].train(True)
        train_avg_loss, train_acc, val_avg_loss, val_acc = train_one_epoch(epoch, train_dataloader, val_dataloader,
                                                                           models[fold])

        print(train_avg_loss)
        print(train_acc)
        print(val_avg_loss)
        print(val_acc)

        losses[fold].append((train_avg_loss, val_avg_loss))
        accuracies[fold].append((train_acc, val_acc))

if __name__ == "__main__":
    models = []
    losses = []
    accuracies = []
    for fold in range(len(train_dataloaders)):
        print('FOLD {}:'.format(fold + 1))

        train_dataloader = train_dataloaders[fold]
        val_dataloader = val_dataloaders[fold]

        model = EncoderClassifier(inputs=xp, embed_size=embed_size, num_heads=num_heads, ff_dim=ff_dim, dropout=dropout,
                                  num_blocks=num_transformer_blocks)
        model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        models.append({"model": model, "loss_fn": criterion, "opt": optimizer})

        losses.append([])
        accuracies.append([])

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch + 1))

            models[fold]["model"].train(True)
            train_avg_loss, train_acc, val_avg_loss, val_acc = train_one_epoch(epoch, train_dataloader, val_dataloader,
                                                                               models[fold])

            print(train_avg_loss)
            print(train_acc)
            print(val_avg_loss)
            print(val_acc)

            losses[fold].append((train_avg_loss, val_avg_loss))
            accuracies[fold].append((train_acc, val_acc))