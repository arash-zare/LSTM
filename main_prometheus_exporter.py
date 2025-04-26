# # main_prometheus_exporter.py
# from flask import Flask, Response
# from prometheus_client import Gauge, generate_latest
# import threading
# import time
# from data_fetcher import fetch_latest_data
# from detect_anomalies import detect_anomaly
# from config import FETCH_INTERVAL

# app = Flask(__name__)
# anomaly_gauge = Gauge('system_anomaly', 'System anomaly detection status (1=anomaly)')
# error_gauge = Gauge('prediction_error', 'Prediction error')

# def monitor():
#     while True:
#         try:
#             data = fetch_latest_data()
#             result, error = detect_anomaly(data)
#             anomaly_gauge.set(result)
#             error_gauge.set(error)
#         except Exception as e:
#             print("Error in detection:", e)
#         time.sleep(FETCH_INTERVAL)

# @app.route("/metrics")
# def metrics():
#     return Response(generate_latest(), mimetype="text/plain")

# if __name__ == "__main__":
#     threading.Thread(target=monitor, daemon=True).start()
#     app.run(host="0.0.0.0", port=8000)



# main_prometheus_exporter.py
from flask import Flask, Response
from prometheus_client import Gauge, generate_latest
import threading
import time
import re
from data_fetcher import fetch_latest_data
from detect_anomalies import detect_anomaly_per_feature
from config import FETCH_INTERVAL, FEATURES

app = Flask(__name__)

# --- Helper function: sanitize full feature names into Prometheus-safe metric names ---
def sanitize_feature_name(feature):
    # فقط حروف کوچک، اعداد و '_' مجازه
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', feature)
    return safe_name.lower()

# --- Create a Gauge per feature ---
anomaly_gauges = {
    feature: Gauge(f"{sanitize_feature_name(feature)}_system_anomaly", f"Anomaly detection for {feature}")
    for feature in FEATURES
}

# --- Monitor function ---
def monitor():
    while True:
        try:
            data = fetch_latest_data()

            anomalies, _ = detect_anomaly_per_feature(data)

            if len(anomalies) != len(FEATURES):
                raise ValueError(f"Mismatch between anomalies ({len(anomalies)}) and features ({len(FEATURES)})")

            for i, feature in enumerate(FEATURES):
                anomaly_gauges[feature].set(anomalies[i])

            print("✅ Updated anomalies:", dict(zip(FEATURES, anomalies)))

        except Exception as e:
            print(f"❌ Error in detection loop: {e}")

        time.sleep(FETCH_INTERVAL)

# --- Flask route for Prometheus ---
@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype="text/plain")

# --- Main ---
if __name__ == "__main__":
    threading.Thread(target=monitor, daemon=True).start()
    print(f"🚀 Starting Prometheus exporter server on port 8000...")
    app.run(host="0.0.0.0", port=8000)
