import os
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer



try:
  from sam import SAM
except ImportError:
    os.system("git clone https://github.com/davda54/sam")
    from sam import sam

try:
    from pytorch_functions import engine_sam, utils
except ImportError:
    os.system("git clone https://github.com/Andresmup/pytorch_functions")
    from pytorch_functions import engine_sam, utils


def train_model_sam(model,
                    train_dataloader,
                    test_dataloader,
                    model_save_name,
                    base_optimizer=torch.optim.Adam,
                    lr=0.001,
                    momentum=0.9,
                    NUM_EPOCHS=5):

    """
    Trains a PyTorch image classification model with SAM.

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

    # Set loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Create base optimizer
    base_optimizer_instance = base_optimizer(model.parameters(), lr=lr, betas=(momentum, 0.999))  # Adjusted betas


    # Create SAM optimizer
    optimizer = SAM(model.parameters(), base_optimizer_instance)

    # Start the timer
    start_time = timer()

    # Train model with SAM
    results = engine_sam.train_sam(
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
