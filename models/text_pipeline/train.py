import torch

from torch.utils.data import DataLoader

from dataset import TextEmotionDataset
from model import TextEmotionModel


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

dataset = TextEmotionDataset("../../data/TESS")

train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, val_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16
)

model = TextEmotionModel().to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=2e-5
)

best_val_loss = float("inf")

patience = 3

counter = 0

epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0
    total = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids,
            attention_mask
        )

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    model.eval()

    val_loss = 0

    with torch.no_grad():

        for batch in val_loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids,
                attention_mask
            )

            loss = criterion(outputs, labels)

            val_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Loss: {running_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_text_model.pth"
        )

        counter = 0

        print("Best model saved!")

    else:

        counter += 1

        print(f"No improvement count: {counter}")

     
print("Training complete!")