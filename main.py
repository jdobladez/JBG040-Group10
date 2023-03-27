# Custom imports
from dc1.batch_sampler import BatchSampler
from dc1.image_dataset import ImageDataset
from dc1.net import Net
from dc1.train_test import train_model, test_model
from early_stop import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# Torch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary  # type: ignore

# Other imports
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.pyplot import figure
import os
import argparse
import plotext  # type: ignore
from datetime import datetime
from pathlib import Path
from typing import List


def main(args: argparse.Namespace, activeloop: bool = True) -> None:

    # Load the train and test data set
    train_dataset = ImageDataset(Path("../data/X_train.npy"), Path("../data/Y_train.npy"))
    test_dataset = ImageDataset(Path("../data/X_test.npy"), Path("../data/Y_test.npy"))

    # Load the Neural Net. NOTE: set number of distinct labels here
    model = Net(n_classes=6)

    # Initialize optimizer(s) and loss function(s)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.1)
    loss_function = nn.CrossEntropyLoss()

    # fetch epoch and batch count from arguments
    n_epochs = args.nb_epochs
    batch_size = args.batch_size

    # IMPORTANT! Set this to True to see actual errors regarding
    # the structure of your model (GPU acceleration hides them)!
    # Also make sure you set this to False again for actual model training
    # as training your model with GPU-acceleration (CUDA/MPS) is much faster.
    DEBUG = False

    # Moving our model to the right device (CUDA will speed training up significantly!)
    if torch.cuda.is_available() and not DEBUG:
        print("@@@ CUDA device found, enabling CUDA training...")
        device = "cuda"
        model.to(device)
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)
    elif (
        torch.backends.mps.is_available() and not DEBUG
    ):  # PyTorch supports Apple Silicon GPU's from version 1.12
        print("@@@ Apple silicon device enabled, training with Metal backend...")
        device = "mps"
        model.to(device)

    else:
        print("@@@ No GPU boosting device found, training on CPU...")
        device = "cpu"
        # Creating a summary of our model and its layers:
        summary(model, (1, 128, 128), device=device)

    # Lets now train, validate and test our model for multiple epochs:
    train_sampler = BatchSampler(
        batch_size=batch_size, dataset=train_dataset, balanced=args.balanced_batches
    )
    test_sampler = BatchSampler(
        batch_size=100, dataset=test_dataset, balanced=args.balanced_batches
    )
    val_sampler = BatchSampler(batch_size=100, dataset=train_dataset, balanced=args.balanced_batches)


    mean_losses_train: List[torch.Tensor] = []
    mean_losses_test: List[torch.Tensor] = []
    mean_losses_val: List[torch.Tensor] = []

    conf_matrix = torch.zeros((6, 6), dtype=torch.int64)

    for e in range(n_epochs):
        if activeloop:

            # Training:
            train_losses = train_model(model, train_sampler, optimizer, loss_function, device)
            # Calculating and printing statistics:
            mean_loss = sum(train_losses) / len(train_losses)
            mean_losses_train.append(mean_loss)
            print(f"\nEpoch {e + 1} training done, loss on train set: {mean_loss}\n")

            # Testing:
            test_losses = test_model(model, test_sampler, loss_function, device)


            y_true = []
            y_pred = []
            for batch in test_sampler:
                X, y = batch
                X = X.to(device)
                y = y.to(device)
                out = model(X)
                pred = out.argmax(dim=1)
                y_true.append(y.cpu())
                y_pred.append(pred.cpu())
                # Update confusion matrix
                conf_matrix += confusion_matrix(y.cpu(), pred.cpu(), labels=[0, 1, 2, 3, 4, 5])

            # Validation:
            val_losses = test_model(model, val_sampler, loss_function, device)
            mean_loss = sum(val_losses) / len(val_losses)
            mean_losses_val.append(mean_loss)
            print(f"\nEpoch {e + 1} validation done, loss on validation set{mean_loss}\n")

            # # Calculating and printing statistics:
            mean_loss = sum(test_losses) / len(test_losses)
            mean_losses_test.append(mean_loss)
            print(f"\nEpoch {e + 1} testing done, loss on test set: {mean_loss}\n")

            ### Plotting during training
            plotext.clf()
            plotext.scatter(mean_losses_train, label="train")
            plotext.scatter(mean_losses_test, label="test")
            plotext.title("Train and test loss")

            plotext.xticks([i for i in range(len(mean_losses_train) + 1)])

            plotext.show()

            # early stopping
            early_stopping = EarlyStopping(tolerance=5, min_delta=10)
            early_stopping(train_losses[e], test_losses[e])
            if early_stopping.early_stop:
                print("We are at epoch:", e)
                # retrieve current time to label artifacts
                now = datetime.now()
                # check if model_weights/ subdir exists
                if not Path("model_weights/").exists():
                    os.mkdir(Path("model_weights/"))

                # Saving the model
                torch.save(model.state_dict(),
                           f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")

                # Create plot of test_losses
                figure(figsize=(9, 10), dpi=80)
                fig, (ax1, ax2) = plt.subplots(2, sharex=True)

                ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train",
                         color="blue")
                ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test",
                         color="red")
                fig.legend()

                # Check if /artifacts/ subdir exists
                if not Path("artifacts/").exists():
                    os.mkdir(Path("artifacts/"))

                # save plot of test_losses
                fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")
                break

    print("Confusion matrix:")
    print(conf_matrix)
    # Plot the confusion matrix
    sns.set(font_scale=1.4)
    sns.heatmap(conf_matrix, annot=True, annot_kws={"size": 16}, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


    # retrieve current time to label artifacts
    now = datetime.now()
    # check if model_weights/ subdir exists
    if not Path("model_weights/").exists():
        os.mkdir(Path("model_weights/"))
    
    # Saving the model
    torch.save(model.state_dict(), f"model_weights/model_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.txt")
    
    # Create plot of test_losses
    figure(figsize=(9, 10), dpi=80)
    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    ax1.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_train], label="Train", color="blue")
    ax2.plot(range(1, 1 + n_epochs), [x.detach().cpu() for x in mean_losses_test], label="Test", color="red")

    fig.legend()
    
    # Check if /artifacts/ subdir exists
    if not Path("artifacts/").exists():
        os.mkdir(Path("artifacts/"))

    # save plot of test_losses
    fig.savefig(Path("artifacts") / f"session_{now.month:02}_{now.day:02}_{now.hour}_{now.minute:02}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--nb_epochs", help="number of training iterations", default=2, type=int
    )
    parser.add_argument("--batch_size", help="batch_size", default=50, type=int)
    parser.add_argument("--val_size", help="size of the validation set", default=0.10, type = float)
    parser.add_argument(
        "--balanced_batches",
        help="whether to balance batches for class labels",
        default=True,
        type=bool,
    )
    args = parser.parse_args()

    main(args)

