import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from torch.utils.data import DataLoader

from dataset import TextEmotionDataset
from model import TextEmotionModel


emotion_names = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "pleasant_surprise",
    "sad"
]

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TextEmotionDataset("../../data/TESS")

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size, test_size]
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16
)

model = TextEmotionModel().to(device)

model.load_state_dict(
    torch.load(
        "best_text_model.pth",
        map_location=device
    )
)

model.eval()

all_preds = []
all_labels = []

correct = 0
total = 0

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids,
            attention_mask
        )

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=emotion_names
    )
)

cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=emotion_names
)

fig, ax = plt.subplots(figsize=(10, 10))

disp.plot(ax=ax)

plt.title("Text Emotion Confusion Matrix")

plt.savefig("../../Results/text_confusion_matrix.png")

plt.show()