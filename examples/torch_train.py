import torch
from torch import optim, nn
import torchvision
import time

class AvgPoolGlobal(nn.Module):
    def forward(self, x):
        return x.mean(axis=(2, 3))

class ResidualBlock(nn.Module):
    def __init__(self, c, d):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(c, d, 3, 1, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.MaxPool2d(3, 1),
            nn.ReLU(),
        )
        self.tail = nn.Sequential(
            nn.Conv2d(d, d, 3, 1, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.head(x)
        y = self.tail(x)
        return x + y

def small_resnet(num_classes):
    head = nn.Sequential(
        nn.Conv2d(3, 32, 3, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(3, 1),
    )
    return nn.Sequential(
        head,
        ResidualBlock(32, 64),
        ResidualBlock(64, 128),
        ResidualBlock(128, 256),
        AvgPoolGlobal(),
        nn.Linear(256, num_classes)
    )

def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    dev = torch.device("cpu")
    model = small_resnet(10)
    opt = optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    train_data = torchvision.datasets.CIFAR10("./datasets", train=True, download=True, transform=torchvision.transforms.ToTensor())
    batch_size = 16
    loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    for i_epoch in range(100):
        for img, lbl in loader:
            img.requires_grad = True
            start = time.perf_counter()
            logits = model(img)
            loss = criterion(logits, lbl)
            fwd_dur = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            opt.zero_grad()
            loss.backward()
            bwd_dur = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            opt.step()
            opt_dur = (time.perf_counter() - start) * 1000

            print(f"{loss.item()} | fwd={fwd_dur:.1f}, bwd={bwd_dur:.1f} opt={opt_dur:.1f}")

if __name__ == "__main__":
    main()