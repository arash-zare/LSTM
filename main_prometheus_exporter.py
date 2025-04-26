

# # main_prometheus_exporter.py
# from flask import Flask, Response
# from prometheus_client import Gauge, generate_latest
# import threading
# import time
# import re
# from data_fetcher import fetch_latest_data
# from detect_anomalies import detect_anomaly_per_feature
# from config import FETCH_INTERVAL, FEATURES

# app = Flask(__name__)

# # --- Helper function: sanitize full feature names into Prometheus-safe metric names ---
# def sanitize_feature_name(feature):
#     # فقط حروف کوچک، اعداد و '_' مجازه
#     safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', feature)
#     return safe_name.lower()

# # --- Create a Gauge per feature ---
# anomaly_gauges = {
#     feature: Gauge(f"{sanitize_feature_name(feature)}_system_anomaly", f"Anomaly detection for {feature}")
#     for feature in FEATURES
# }

# # --- Monitor function ---
# def monitor():
#     while True:
#         try:
#             data = fetch_latest_data()

#             anomalies, _ = detect_anomaly_per_feature(data)

#             if len(anomalies) != len(FEATURES):
#                 raise ValueError(f"Mismatch between anomalies ({len(anomalies)}) and features ({len(FEATURES)})")

#             for i, feature in enumerate(FEATURES):
#                 anomaly_gauges[feature].set(anomalies[i])

#             print("✅ Updated anomalies:", dict(zip(FEATURES, anomalies)))

#         except Exception as e:
#             print(f"❌ Error in detection loop: {e}")

#         time.sleep(FETCH_INTERVAL)

# # --- Flask route for Prometheus ---
# @app.route("/metrics")
# def metrics():
#     return Response(generate_latest(), mimetype="text/plain")

# # --- Main ---
# if __name__ == "__main__":
#     threading.Thread(target=monitor, daemon=True).start()
#     print(f"🚀 Starting Prometheus exporter server on port 8000...")
#     app.run(host="0.0.0.0", port=8000)




# # main_prometheus_exporter.py
# from flask import Flask, Response
# from prometheus_client import Gauge, generate_latest
# import threading
# import time
# import re
# import torch
# import numpy as np
# from data_fetcher import fetch_latest_data
# from detect_anomalies import detect_anomaly_per_feature
# from model import LSTMAnomalyDetector
# from config import FETCH_INTERVAL, FEATURES, MODEL_PATH, SEQUENCE_LENGTH

# app = Flask(__name__)

# # --- Helper function: sanitize full feature names into Prometheus-safe metric names ---
# def sanitize_feature_name(feature):
#     # فقط حروف کوچک، اعداد و '_' مجازه
#     safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', feature)
#     return safe_name.lower()

# # --- Create a Gauge per feature ---
# anomaly_gauges = {
#     feature: Gauge(f"{sanitize_feature_name(feature)}_system_anomaly", f"Anomaly detection for {feature}")
#     for feature in FEATURES
# }

# # --- Load trained model ---
# model = LSTMAnomalyDetector()
# model.load_state_dict(torch.load(MODEL_PATH))
# model.eval()
# print(f"✅ Loaded model from {MODEL_PATH}")

# # --- Internal buffer to hold sequences ---
# sequence_buffer = []

# # --- Monitor function ---
# def monitor():
#     global sequence_buffer

#     while True:
#         try:
#             # Fetch latest data point
#             latest_data = fetch_latest_data()
#             if len(latest_data) != len(FEATURES):
#                 raise ValueError(f"Mismatch between fetched data ({len(latest_data)}) and features ({len(FEATURES)})")

#             sequence_buffer.append(latest_data)

#             # Only start detecting when enough data is collected
#             if len(sequence_buffer) >= SEQUENCE_LENGTH:
#                 # Prepare input
#                 input_sequence = np.array(sequence_buffer[-SEQUENCE_LENGTH:])  # فقط آخرین sequence
#                 input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)  # batch dimension

#                 # Predict
#                 with torch.no_grad():
#                     output = model(input_tensor)
                
#                 predicted = output.squeeze(0).numpy()  # (sequence_length, num_features)

#                 # فقط آخرین لحظه رو مقایسه کنیم
#                 actual_last = np.array(sequence_buffer[-1])
#                 predicted_last = predicted[-1]

#                 # تشخیص آنومالی بر اساس اختلاف
#                 anomalies = np.abs(actual_last - predicted_last)
#                 anomalies = anomalies.tolist()

#                 for i, feature in enumerate(FEATURES):
#                     anomaly_gauges[feature].set(anomalies[i])

#                 print("✅ Updated anomalies:", dict(zip(FEATURES, anomalies)))

#         except Exception as e:
#             print(f"❌ Error in detection loop: {e}")

#         time.sleep(FETCH_INTERVAL)

# # --- Flask route for Prometheus ---
# @app.route("/metrics")
# def metrics():
#     return Response(generate_latest(), mimetype="text/plain")

# # --- Main ---
# if __name__ == "__main__":
#     threading.Thread(target=monitor, daemon=True).start()
#     print(f"🚀 Starting Prometheus exporter server on port 8000...")
#     app.run(host="0.0.0.0", port=8000)



# main_prometheus_exporter.py
from flask import Flask, Response
from prometheus_client import Gauge, generate_latest, REGISTRY
import threading
import time
import re
import torch
import numpy as np
from data_fetcher import fetch_latest_data
from detect_anomalies import detect_anomaly_per_feature
from model import LSTMAnomalyDetector
# from config import FETCH_INTERVAL, FEATURES, MODEL_PATH, SEQUENCE_LENGTH, THRESHOLD
from config import FETCH_INTERVAL, FEATURES, MODEL_PATH, SEQUENCE_LENGTH, THRESHOLDS

app = Flask(__name__)

# --- Helper function: sanitize feature names ---
def sanitize_feature_name(feature):
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', feature)
    return safe_name.lower()

# --- Remove previous metrics if exist ---
def safe_gauge(name, documentation):
    try:
        metric = Gauge(name, documentation, registry=None)
        REGISTRY.unregister(metric)
    except KeyError:
        pass
    return Gauge(name, documentation)

# --- Create Gauges per feature ---
anomaly_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_system_anomaly", f"Anomaly detection for {feature}")
    for feature in FEATURES
}

mse_gauges = {
    feature: safe_gauge(f"{sanitize_feature_name(feature)}_system_MSE", f"MSE error for {feature}")
    for feature in FEATURES
}

# --- Load trained model ---
model = LSTMAnomalyDetector()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print(f"✅ Loaded model from {MODEL_PATH}")

# --- Internal buffer ---
sequence_buffer = []

# --- Monitor function ---
def monitor():
    global sequence_buffer

    while True:
        try:
            # Fetch latest data point
            latest_data = fetch_latest_data()
            if len(latest_data) != len(FEATURES):
                raise ValueError(f"Mismatch between fetched data ({len(latest_data)}) and features ({len(FEATURES)})")

            sequence_buffer.append(latest_data)

            # Keep only needed sequence length
            if len(sequence_buffer) > SEQUENCE_LENGTH:
                sequence_buffer = sequence_buffer[-SEQUENCE_LENGTH:]

            if len(sequence_buffer) == SEQUENCE_LENGTH:
                # Prepare input
                input_sequence = np.array(sequence_buffer)
                input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)  # batch=1

                # Predict
                with torch.no_grad():
                    output = model(input_tensor)

                predicted = output.squeeze(0).numpy()

                # Compare last timestep
                actual_last = np.array(sequence_buffer[-1])
                predicted_last = predicted[-1]

                # Compute MSE per feature
                mse_per_feature = (actual_last - predicted_last) ** 2

                # Set metrics
                for i, feature in enumerate(FEATURES):
                    mse_value = mse_per_feature[i]
                    mse_gauges[feature].set(mse_value)

                    # Set anomaly: 1 if mse > threshold, else 0
                    threshold = THRESHOLDS[feature]
                    is_anomaly = 1 if mse_value > threshold else 0

                    # is_anomaly = 1 if mse_value > THRESHOLD else 0
                    anomaly_gauges[feature].set(is_anomaly)

                print("✅ Updated metrics:", dict(zip(FEATURES, mse_per_feature)))

        except Exception as e:
            print(f"❌ Error in detection loop: {e}")

        time.sleep(FETCH_INTERVAL)

# --- Flask route ---
@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

# --- Main ---
if __name__ == "__main__":
    threading.Thread(target=monitor, daemon=True).start()
    print(f"🚀 Starting victoriametric server on port 8000...")
    app.run(host="0.0.0.0", port=8000)
