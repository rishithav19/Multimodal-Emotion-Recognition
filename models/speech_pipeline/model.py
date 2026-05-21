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

            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Fully Connected Layers
        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):

        # CNN
        x = self.cnn(x)

        # Convert for LSTM
        x = x.permute(0, 2, 1)

        # BiLSTM
        output, (hidden, cell) = self.lstm(x)

        # Concatenate forward + backward hidden states
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        x = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        # Classification
        x = self.classifier(x)

        return x