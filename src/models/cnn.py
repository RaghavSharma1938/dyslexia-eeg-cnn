import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, n_ch, n_classes=2, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch,  64, 25, 2, 12), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64,   128, 15, 2, 7 ), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,  256,  9, 2, 4 ), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256,  256,  7, 2, 3 ), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(p_drop),
            nn.Linear(256, n_classes)
        )
    def forward(self, x): return self.net(x)
