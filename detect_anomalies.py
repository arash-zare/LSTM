# # detect_anomalies.py
# import torch
# from model import load_model
# from preprocessing import preprocess_input
# from config import THRESHOLD

# model = load_model()

# def detect_anomaly(data):
#     x = torch.tensor(preprocess_input(data), dtype=torch.float32).unsqueeze(0)
#     pred = model(x)
#     error = torch.mean((x[:, -1, :] - pred) ** 2).item()
#     return 1 if error > THRESHOLD else 0, error



# # detect_anomalies.py
# import torch
# from model import load_model
# from preprocessing import preprocess_input
# from config import THRESHOLD, FEATURES

# model = load_model()

# def detect_anomaly_per_feature(data):
#     x = torch.tensor(preprocess_input(data), dtype=torch.float32).unsqueeze(0)
#     pred = model(x)
#     errors = ((x[:, -1, :] - pred) ** 2).squeeze(0).tolist()
    
#     anomalies = [1 if err > THRESHOLD else 0 for err in errors]
#     return anomalies, errors



# detect_anomalies.py
import torch
import numpy as np
from model import load_model
from preprocessing import fit_scaler, preprocess_input
from config import THRESHOLD

model = load_model()

def detect_anomaly_per_feature(raw_data_batch):
    """
    raw_data_batch: list or numpy array with shape (features,)
    Example: [cpu_value, ram_value]
    """
    raw_data_batch = np.array(raw_data_batch).reshape(1, -1)  # 🛠️ تبدیل به (1, features)

    # 🛠️ اول fit scaler روی همین batch
    fit_scaler(raw_data_batch)

    # 🎛️ بعدش تبدیل به input
    x_preprocessed = preprocess_input(raw_data_batch)
    x_tensor = torch.tensor(x_preprocessed, dtype=torch.float32)  # (1, features)

    x_tensor = x_tensor.unsqueeze(0)  # 🔥 تبدیل به (batch_size=1, sequence_length=1, features)

    # 🔮 پیش‌بینی
    pred = model(x_tensor)

    # 📈 محاسبه خطا
    errors = ((x_tensor[:, -1, :] - pred) ** 2).squeeze(0).tolist()

    # 🚨 تصمیم نهایی
    anomalies = [1 if err > THRESHOLD else 0 for err in errors]

    return anomalies, errors
