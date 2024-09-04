from torch import nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.relu = nn.ReLU()
        self.layer = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, )
        self.pooling = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(984, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 5)
        self.embedding = nn.Embedding(6, 8)

    # @torchsnooper.snoop()
    def forward(self, x, **kwargs):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x
