import os
import glob
import torch

from torch.utils.data import Dataset
from transformers import BertTokenizer


EMOTIONS = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "ps": 5,
    "sad": 6
}


class TextEmotionDataset(Dataset):

    def __init__(self, root_path):

        self.files = glob.glob(
            os.path.join(root_path, "**", "*.wav"),
            recursive=True
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )

    def get_text(self, path):

        filename = os.path.basename(path)

        parts = filename.split("_")

        word = parts[1]

        return word.lower()

    def get_label(self, path):

        filename = os.path.basename(path).lower()

        for emotion in EMOTIONS:

            if emotion in filename:
                return EMOTIONS[emotion]

        return 0

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        path = self.files[idx]

        text = self.get_text(path)

        label = self.get_label(path)

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=10,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label)
        }