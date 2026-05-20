import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from torch.utils.data import DataLoader

from dataset import TESSDataset
from model import SpeechEmotionModel


# Emotion Labels
emotion_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad"
]

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
dataset = TESSDataset("../../data/TESS")

# Split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, test_size]
)

# Test Loader
test_loader = DataLoader(
    test_dataset,
    batch_size=16
)

# Load Model
model = SpeechEmotionModel().to(device)

model.load_state_dict(
    torch.load(
        "speech_emotion_model.pth",
        map_location=device
    )
)

model.eval()

all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for features, labels in test_loader:

        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Accuracy
accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

# Classification Report
print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=emotion_names
    )
)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=emotion_names
)

fig, ax = plt.subplots(figsize=(10, 10))

disp.plot(ax=ax)

plt.title("Speech Emotion Confusion Matrix")

plt.show()