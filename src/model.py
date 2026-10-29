import torch
from torch import nn

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 10))

    def forward(self, x, verbose=False):
        if verbose: print("Entrada:", tuple(x.shape))
        x = self.block1(x)
        if verbose: print("Bloque 1:", tuple(x.shape))
        x = self.block2(x)
        if verbose: print("Bloque 2:", tuple(x.shape))
        x = self.block3(x)
        if verbose: print("Bloque 3:", tuple(x.shape))
        x = self.classifier(x)
        if verbose: print("Salida:", tuple(x.shape))
        return x

def validate_shapes():
    model = CIFAR10CNN()
    dummy = torch.randn(8, 3, 32, 32)
    with torch.no_grad():
        result = model(dummy, verbose=True)
    assert result.shape == (8, 10), f"Shape incorrecto: {result.shape}"
    print("Aserción correcta: (8, 10)")

if __name__ == "__main__":
    validate_shapes()
