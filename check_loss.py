import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def load_loss(ckpt_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False) # load current model
    # weights_only=False is required on pytorch 2.6

    epoch_losses = checkpoint["epoch_losses"]
    epoch_val_losses = checkpoint["epoch_val_losses"]
    best_loss = checkpoint["best_loss"]
    epoch_accs = checkpoint["epoch_accs"]
    epoch_val_accs = checkpoint["epoch_val_accs"]
    print(best_loss)
    print(epoch_val_losses)
    return epoch_losses, epoch_val_losses, epoch_accs, epoch_val_accs

def visualize(losses: list, save_dir: str, mode: str="val"):
    epochs = np.arange(1, len(losses) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker='o', color='blue', label='Loss')
    if mode=="train":
        plt.title('Train Loss Curve per Epoch')
    elif mode=="val":
        plt.title('Val Loss Curve per Epoch')
    else:
        plt.title('Loss Curve per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{mode}_loss.png")
    plt.close()

def visualize_losses(train_losses: list, val_losses: list, save_dir: str):
    # losses
    train_epochs = np.arange(1, len(train_losses)+1)
    val_epochs = np.arange(1, len(val_losses)+1)
    plt.figure(figsize=(8, 5))
    plt.plot(train_epochs, train_losses, marker='o', color='blue', label='Train Loss')
    plt.plot(val_epochs, val_losses, marker='o', color='orange', label='Validation Loss')
    plt.title('Train and Validation Loss Curve per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/train_val_loss.png")
    plt.savefig(f"{save_dir}/train_val_loss.eps")
    plt.close()

def visualize_accs(train_accs: list, val_accs: list, save_dir: str):
    # losses
    train_epochs = np.arange(1, len(train_accs)+1)
    val_epochs = np.arange(1, len(val_accs)+1)
    plt.figure(figsize=(8, 5))
    plt.plot(train_epochs, train_accs, marker='o', color='blue', label='Train Accuracy')
    plt.plot(val_epochs, val_accs, marker='o', color='orange', label='Validation Accuracy')
    plt.title('Train and Validation Accuracy Curve per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0,1])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/train_val_acc.png")
    plt.savefig(f"{save_dir}/train_val_acc.eps")
    plt.close()


def main():
    ckpt_path = "models/100_0.648.pt" # "models/100_1.267.pt"
    dir_path = os.path.dirname(ckpt_path)
    train_losses, val_losses, train_accs, val_accs = load_loss(ckpt_path=ckpt_path)
    # print(train_losses, len(train_losses))
    # visualize(losses=train_losses, save_dir=dir_path, mode="train")
    # visualize(losses=val_losses, save_dir=dir_path)
    visualize_losses(train_losses=train_losses, val_losses=val_losses, save_dir=dir_path)
    visualize_accs(train_accs=train_accs, val_accs=val_accs, save_dir=dir_path)

if __name__ == "__main__":
    main()