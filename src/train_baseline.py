# Train a simple CNN from scratch to establish performance baseline
#  before using a pretrained model

# Train a simple CNN -> Measure performance -> Use it as reference point

# Trained a baseline CNN from scratch to understand dataset difficulty and
# to justify the use of transfer learning

# Baseline CNN training code below
import torch
import torch.nn as nn


# model definition
class FashionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


# for inference
def load_model():
    model = FashionNet()
    model.load_state_dict(
        torch.load("/app/model.pt", map_location="cpu")
    )
    model.eval()
    return model
