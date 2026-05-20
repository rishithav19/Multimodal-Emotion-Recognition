import librosa
import librosa.display
import matplotlib.pyplot as plt

path = "data/TESS/OAF_angry/OAF_back_angry.wav"

audio, sr = librosa.load(path, sr=22050)

# Extract MFCC
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

print("MFCC Shape:", mfcc.shape)

# Visualize
plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mfcc,
    x_axis='time'
)

plt.colorbar()
plt.title("MFCC Features")
plt.tight_layout()

plt.show()