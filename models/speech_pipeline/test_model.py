import torch

from model import SpeechEmotionModel

model = SpeechEmotionModel()

dummy_input = torch.randn(
    8,      # batch size
    40,     # MFCC features
    200     # time steps
)

output = model(dummy_input)

print(output.shape)