import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataset import TESSDataset
from model import SpeechEmotionModel


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)


dataset = TESSDataset("../../data/TESS")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16
)

model = SpeechEmotionModel().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


epochs = 20

for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )

# Save Model
torch.save(
    model.state_dict(),
    "speech_emotion_model.pth"
)

print("Model saved!")