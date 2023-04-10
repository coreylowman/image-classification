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

    dev = torch.device("cuda")
    model = small_resnet(10).to(dev)
    opt = optim.SGD(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    train_data = torchvision.datasets.CIFAR10("./datasets", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.CIFAR10("./datasets", train=False, download=True, transform=torchvision.transforms.ToTensor())
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for i_epoch in range(100):
        for img, lbl in train_loader:
            # img.requires_grad = True
            start = time.perf_counter()
            logits = model(img.to(dev))
            loss = criterion(logits, lbl.to(dev))
            # torch.cuda.synchronize()
            fwd_dur = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            opt.zero_grad()
            loss.backward()
            # torch.cuda.synchronize()
            bwd_dur = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            opt.step()
            # torch.cuda.synchronize()
            opt_dur = (time.perf_counter() - start) * 1000

            # print(f"{loss.item()} | fwd={fwd_dur:.1f}, bwd={bwd_dur:.1f} opt={opt_dur:.1f}")

        num_correct = 0
        num_total = 0
        for img, lbl in test_loader:
            probs = model(img.to(dev)).softmax(-1)
            num_correct += (probs.max(-1)[1] == lbl.to(dev)).sum().item()
            num_total += img.shape[0]
        print(num_correct / num_total)

if __name__ == "__main__":
    main()