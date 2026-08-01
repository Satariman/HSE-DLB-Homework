from torch import nn
from torch.nn import Sequential

from src.transforms.max_feature_map import MaxFeatureMap

class LCNNModel(nn.Module):

    def __init__(self, in_channels, dropout_probability):
        super().__init__()

        self.net = Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=5, stride=1),
            MaxFeatureMap(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=96, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap(),

            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),

            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1, stride=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(48),
            nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1),
            MaxFeatureMap(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            MaxFeatureMap(),

            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(4416, 160),
            MaxFeatureMap(dim=-1),
            nn.Dropout(dropout_probability),
            nn.BatchNorm1d(80),

            nn.Linear(80, 2)
        )

    def forward(self, data_object, **batch):

        return {"logits": self.net(data_object)}

    def __str__(self):

        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


