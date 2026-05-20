import librosa

path = "data/TESS/OAF_angry/OAF_back_angry.wav"

audio, sr = librosa.load(path, sr=22050)

print(audio.shape)
print(sr)