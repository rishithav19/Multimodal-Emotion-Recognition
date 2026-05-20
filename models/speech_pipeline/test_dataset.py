from dataset import TESSDataset
dataset = TESSDataset("../../data/TESS")

print("Dataset size:", len(dataset))

x, y = dataset[0]

print("Feature shape:", x.shape)
print("Label:", y)