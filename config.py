# # config.py
# VICTORIA_METRICS_URL = "http://192.168.1.98:8428/api/v1/query"
# FEATURES = ["node_cpu_seconds_total"]
# # "node_memory_Active_bytes"
# THRESHOLD = 0.01  # Mean squared error threshold for anomaly
# SEQUENCE_LENGTH = 10
# INPUT_DIM = len(FEATURES)
# HIDDEN_DIM = 64
# NUM_LAYERS = 2
# MODEL_PATH = "lstm_model.pth"
# FETCH_INTERVAL = 60  # seconds


# config.py
VICTORIA_METRICS_URL = "http://192.168.1.98:8428/api/v1/query"
FEATURES = [
    "node_cpu_seconds_total",
    "node_network_receive_packets_total",
    "node_network_transmit_packets_total",
    "node_nf_conntrack_entries",
    "node_network_receive_bytes_total",
    "node_network_transmit_bytes_total"
]
THRESHOLD = 0.01  # Mean squared error threshold for anomaly
SEQUENCE_LENGTH = 10
INPUT_DIM = len(FEATURES)
HIDDEN_DIM = 64
NUM_LAYERS = 2
MODEL_PATH = "lstm_model.pth"
FETCH_INTERVAL = 60  # seconds
