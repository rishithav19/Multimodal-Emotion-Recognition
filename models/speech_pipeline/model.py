import torch
import torch.nn as nn


class SpeechEmotionModel(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(

            nn.Conv1d(
                in_channels=40,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool1d(kernel_size=2)
        )

        # LSTM Temporal Modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        # Classifier
        self.fc = nn.Linear(
            128,
            num_classes
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        output, (hidden, cell) = self.lstm(x)
        x = hidden[-1]
        x = self.fc(x)
        return x