"""
Contains functions for training and testing a PyTorch model.
"""
from typing import Dict, List, Tuple

import torch

from tqdm.auto import tqdm

def train_sam(model,
              train_dataloader,
              test_dataloader,
              optimizer,
              loss_fn,
              epochs,
              device):
    
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step_sam(model=model,
                                               dataloader=train_dataloader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               device=device)
        test_loss, test_acc = test_step_sam(model=model,
                                            dataloader=test_dataloader,
                                            loss_fn=loss_fn,
                                            device=device)

        # Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return results

def train_step_sam(model,
                   dataloader,
                   loss_fn,
                   optimizer,
                   device):
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # First forward-backward pass
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # Second forward-backward pass
        loss_fn(model(X), y).backward()
        optimizer.second_step(zero_grad=True)

        # Calculate and accumulate loss metric across all batches
        train_loss += loss.item()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(model(X), dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred_class)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step_sam(model,
                  dataloader,
                  loss_fn,
                  optimizer,  # Necesitamos el optimizador para el segundo paso
                  device):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Loop through DataLoader batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # First forward-backward pass
        loss = loss_fn(model(X), y)
        optimizer.first_step(zero_grad=True)

        # Second forward-backward pass
        loss_fn(model(X), y).backward()
        optimizer.second_step(zero_grad=True)  # Utilizar el segundo paso para la optimización

        # Calculate and accumulate accuracy
        test_pred_labels = torch.argmax(torch.softmax(model(X), dim=1), dim=1)
        test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc
