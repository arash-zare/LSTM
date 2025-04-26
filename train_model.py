# # train_model.py

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from model import LSTMAnomalyDetector
# from config import MODEL_PATH, FEATURES

# # مدل رو تعریف کن
# input_size = len(FEATURES)
# hidden_size = 64
# # model = LSTMAnomalyDetector()
# model = LSTMAnomalyDetector(input_dim=input_size, hidden_dim=hidden_size, num_layers=2)




# # Loss function و Optimizer
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # ساختن دیتا آموزشی ساده (داده سالم فرضی)
# np.random.seed(42)
# data = np.random.normal(loc=0.5, scale=0.05, size=(1000, len(FEATURES)))  # 1000 نمونه نرمال
# data = torch.tensor(data, dtype=torch.float32)

# # آماده سازی دیتاست برای LSTM
# sequence_length = 10
# x_train = []
# y_train = []

# for i in range(len(data) - sequence_length):
#     x_seq = data[i:i+sequence_length]
#     y_seq = data[i+1:i+sequence_length+1]
#     x_train.append(x_seq)
#     y_train.append(y_seq)

# x_train = torch.stack(x_train)
# y_train = torch.stack(y_train)

# # آموزش مدل
# num_epochs = 20

# for epoch in range(num_epochs):
#     model.train()
#     output = model(x_train)
#     loss = criterion(output, y_train[:, -1, :])  # توجه به خروجی fc روی آخرین لحظه
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# # ذخیره مدل
# torch.save(model.state_dict(), MODEL_PATH)
# print(f"✅ Model saved to {MODEL_PATH}")



# train_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import LSTMAnomalyDetector
from config import MODEL_PATH, FEATURES, SEQUENCE_LENGTH

# تعریف مدل
input_size = len(FEATURES)
hidden_size = 64
model = LSTMAnomalyDetector()

# Loss function و Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ساخت دیتا آموزشی سالم (فرضی)
np.random.seed(42)
data = np.random.normal(loc=0.5, scale=0.05, size=(1000, input_size))  # 1000 نمونه سالم
data = torch.tensor(data, dtype=torch.float32)

# آماده‌سازی دیتاست برای LSTM
sequence_length = SEQUENCE_LENGTH
x_train = []
y_train = []

for i in range(len(data) - sequence_length):
    x_seq = data[i:i+sequence_length]
    y_seq = data[i+1:i+sequence_length+1]
    x_train.append(x_seq)
    y_train.append(y_seq)

x_train = torch.stack(x_train)
y_train = torch.stack(y_train)

# آموزش مدل
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    output = model(x_train)
    loss = criterion(output, y_train[:, -1, :])  # فقط خروجی آخرین لحظه
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# ذخیره مدل
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
