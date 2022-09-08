from torch import nn


class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        channels: int = 3, 
        width: int = 32, 
        height: int = 32, 
        num_classes: int = 10, 
        hidden_size: int = 64, 
        learning_rate: float = 2e-4,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.net(x)


if __name__ == "__main__":
    _ = SimpleDenseNet()
