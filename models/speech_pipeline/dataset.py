import os
import glob
import librosa
import numpy as np
import torch

from torch.utils.data import Dataset

EMOTIONS = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "ps": 5,
    "sad": 6
}


class TESSDataset(Dataset):

    def __init__(self, root_path):

        self.files = glob.glob(
            os.path.join(root_path, "**", "*.wav"),
            recursive=True
        )

    def extract_features(self, path):

        audio, sr = librosa.load(path, sr=22050)

        # Trim silence
        audio, _ = librosa.effects.trim(audio)

        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40
        )

        max_len = 200

        if mfcc.shape[1] < max_len:

            pad_width = max_len - mfcc.shape[1]

            mfcc = np.pad(
                mfcc,
                pad_width=((0, 0), (0, pad_width)),
                mode='constant'
            )

        else:
            mfcc = mfcc[:, :max_len]

        return mfcc

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

        features = self.extract_features(path)

        label = self.get_label(path)

        features = torch.tensor(
            features,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return features, label