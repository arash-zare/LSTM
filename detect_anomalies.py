
# # detect_anomalies.py
# import torch
# import numpy as np
# from model import load_model
# from preprocessing import fit_scaler, preprocess_input
# from config import THRESHOLD

# model = load_model()

# def detect_anomaly_per_feature(raw_data_batch):
#     """
#     raw_data_batch: list or numpy array with shape (features,)
#     Example: [cpu_value, ram_value]
#     """
#     raw_data_batch = np.array(raw_data_batch).reshape(1, -1)  # 🛠️ تبدیل به (1, features)

#     # 🛠️ اول fit scaler روی همین batch
#     fit_scaler(raw_data_batch)

#     # 🎛️ بعدش تبدیل به input
#     x_preprocessed = preprocess_input(raw_data_batch)
#     x_tensor = torch.tensor(x_preprocessed, dtype=torch.float32)  # (1, features)

#     x_tensor = x_tensor.unsqueeze(0)  # 🔥 تبدیل به (batch_size=1, sequence_length=1, features)

#     # 🔮 پیش‌بینی
#     pred = model(x_tensor)

#     # 📈 محاسبه خطا
#     errors = ((x_tensor[:, -1, :] - pred) ** 2).squeeze(0).tolist()

#     # 🚨 تصمیم نهایی
#     anomalies = [1 if err > THRESHOLD else 0 for err in errors]

#     return anomalies, errors



# detect_anomalies.py
import torch
import numpy as np
from model import load_model
from preprocessing import fit_scaler, preprocess_input
from config import THRESHOLDS, FEATURES  # 🛠️ اینجا باید THRESHOLDS بیاری نه THRESHOLD

model = load_model()

def detect_anomaly_per_feature(raw_data_batch):
    """
    raw_data_batch: list or numpy array with shape (features,)
    Example: [cpu_value, ram_value, ...]
    """
    raw_data_batch = np.array(raw_data_batch).reshape(1, -1)  # 🛠️ (1, features)

    # 🛠️ Fit scaler روی همین batch
    fit_scaler(raw_data_batch)

    # 🎛️ Preprocess input
    x_preprocessed = preprocess_input(raw_data_batch)
    x_tensor = torch.tensor(x_preprocessed, dtype=torch.float32)

    x_tensor = x_tensor.unsqueeze(0)  # (batch_size=1, sequence_length=1, features)

    # 🔮 Predict
    pred = model(x_tensor)

    # 📈 Calculate MSE error
    errors = ((x_tensor[:, -1, :] - pred) ** 2).squeeze(0).tolist()

    # 🚨 Decide anomalies per feature
    anomalies = []
    for i, feature in enumerate(FEATURES):
        threshold = THRESHOLDS.get(feature, 1e9)  # اگه فیچری نبود، مقدار خیلی بزرگ بذار
        is_anomaly = 1 if errors[i] > threshold else 0
        anomalies.append(is_anomaly)

    return anomalies, errors
