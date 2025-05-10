import torch.nn as nn


import torch.nn as nn

class SmokingCNN(nn.Module):
    def __init__(self, version: int = 1):
        super(SmokingCNN, self).__init__()

        if version == 1:
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Flatten(),

                nn.Linear(64 * 16 * 16, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        elif version == 2:
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Flatten(),

                nn.Linear(128 * 8 * 8, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        
        elif version == 3:
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Flatten(),

                nn.Linear(256 * 4 * 4, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        elif version == 4:
            self.model = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                nn.Flatten(),

                nn.Linear(64 * 16 * 16, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 1)
            )
        else:
            raise ValueError("Invalid version number. Choose between 1 and 4.")

    def forward(self, x):
        return self.model(x)
