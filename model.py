# # model.py
# import torch
# import torch.nn as nn
# from config import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, MODEL_PATH

# class LSTMAnomalyDetector(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
#         self.fc = nn.Linear(HIDDEN_DIM, INPUT_DIM)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.fc(out[:, -1, :])

# def load_model():
#     model = LSTMAnomalyDetector()
#     model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
#     model.eval()
#     return model


# model.py
import torch
import torch.nn as nn
from config import INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, MODEL_PATH

class LSTMAnomalyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, INPUT_DIM)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # فقط آخرین تایم‌استپ
        out = self.fc(out)
        return out

def load_model():
    model = LSTMAnomalyDetector()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model
