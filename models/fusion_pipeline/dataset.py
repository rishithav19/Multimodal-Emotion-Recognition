import os
import glob
import librosa
import numpy as np
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


class FusionDataset(Dataset):

    def __init__(self, root_path):

        self.files = glob.glob(
            os.path.join(root_path, "**", "*.wav"),
            recursive=True
        )

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased"
        )

    def extract_mfcc(self, path):

        signal, sr = librosa.load(
            path,
            sr=22050
        )

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=40
        )

        if mfcc.shape[1] < 200:

            pad_width = 200 - mfcc.shape[1]

            mfcc = np.pad(
                mfcc,
                pad_width=((0, 0), (0, pad_width))
            )

        else:

            mfcc = mfcc[:, :200]

        return torch.tensor(
            mfcc,
            dtype=torch.float32
        )

    def get_text(self, path):

        filename = os.path.basename(path)

        parts = filename.split("_")

        return parts[1].lower()

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

        speech_features = self.extract_mfcc(path)

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
            "speech": speech_features,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label)
        }