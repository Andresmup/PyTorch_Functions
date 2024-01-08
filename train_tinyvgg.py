import os
import torch
import argparse
import data_setup
import engine
import model_builder
import utils
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PyTorch image classification model.")
    parser.add_argument("--model", type=str, default="TinyVGG", help="Name of the model (default: TinyVGG)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs (default: 5)")
    parser.add_argument("--name_saving", type=str, default="model", help="Name for saving the model (default: model)")

    return parser.parse_args()

def main():
    args = parse_args()

    # Setup directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"

    # Setup target device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create transforms
    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Create transforms
    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Create DataLoaders
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        train_transform=train_transforms,
        test_transform=test_transforms,
        batch_size=args.batch_size
    )

    # Create model
    model = tinyvgg_model_builder.TinyVGG(input_shape=3,hidden_units=10,output_shape=len(class_names)).to(device)

    # Set loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Start training
    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.num_epochs,
        device=device
    )

    # Save the model
    utils.save_model(
        model=model,
        target_dir="models",
        model_name=f"{args.name_saving}_model.pth"
    )

if __name__ == "__main__":
    main()
