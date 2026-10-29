import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SEED = 59
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_loaders(batch_size=128, root="./data"):
    set_seed()
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    base = datasets.CIFAR10(root=root, train=True, download=True)
    idx = np.arange(len(base))
    tr_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=np.array(base.targets))
    train_data = Subset(datasets.CIFAR10(root=root, train=True, download=False, transform=train_tf), tr_idx)
    val_data = Subset(datasets.CIFAR10(root=root, train=True, download=False, transform=eval_tf), val_idx)
    test_data = datasets.CIFAR10(root=root, train=False, download=True, transform=eval_tf)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Semilla: {SEED}; train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}")
    print(f"Batch train X: {next(iter(train_loader))[0].shape}")
    return train_loader, val_loader, test_loader

def show_samples(root="./data", save_path=None):
    set_seed()
    data = datasets.CIFAR10(root=root, train=True, download=True)
    indices = np.random.choice(len(data), 16, replace=False)
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for ax, idx in zip(axes.flat, indices):
        img, label = data[int(idx)]
        ax.imshow(img)
        ax.set_title(CLASSES[label])
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
    plt.show()

if __name__ == "__main__":
    show_samples()
    get_loaders()
