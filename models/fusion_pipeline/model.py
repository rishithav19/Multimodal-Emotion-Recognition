import torch
import torch.nn as nn

from transformers import BertModel


class FusionEmotionModel(nn.Module):

    def __init__(self, num_classes=7):

        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv1d(
                40,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        self.fusion = nn.Sequential(

            nn.Linear(256 + 768, 256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(
        self,
        speech,
        input_ids,
        attention_mask
    ):

        speech = self.cnn(speech)

        speech = speech.permute(0, 2, 1)

        _, (hidden, _) = self.lstm(speech)

        speech_embedding = torch.cat(
            (hidden[-2], hidden[-1]),
            dim=1
        )

        text_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_embedding = text_outputs.pooler_output

        combined = torch.cat(
            (speech_embedding, text_embedding),
            dim=1
        )

        output = self.fusion(combined)

        return output