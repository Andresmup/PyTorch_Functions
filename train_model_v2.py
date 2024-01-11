import os
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer

try:
    from pytorch_functions import engine, utils
except ImportError:
    os.system("git clone https://github.com/Andresmup/pytorch_functions")
    from pytorch_functions import engine, utils


def train_model(model, 
                train_dataloader, 
                test_dataloader, 
                model_save_name, 
                optimizer=None,
                NUM_EPOCHS=5, 
                LEARNING_RATE=0.001):
    """
    Trains a PyTorch image classification model.

    Args:
        model: PyTorch model to be trained.
        train_dataloader: torch DataLoaders with training data.
        test_dataloader: torch DataLoaders with testing data.
        model_save_name: Name to save the trained model.
        optimizer: PyTorch optimizer (optional, default is None).
        NUM_EPOCHS: Number of training epochs (default is 5).
        LEARNING_RATE: Learning rate for the optimizer (default is 0.001).
    """
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Start the timer
    start_time = timer()

    # Train model
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device
    )

    # End the timer and print out how long it took
    end_time = timer()
    execution_time = end_time - start_time
    print(f"[INFO] Total training time: {execution_time:.3f} seconds")

    # Save the model
    utils.save_model(
        model=model,
        target_dir="models",
        model_name=model_save_name
    )
    return results, execution_time
