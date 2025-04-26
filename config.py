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


THRESHOLDS = {
    "node_cpu_seconds_total": 4e+10,
    "node_network_receive_packets_total": 1.23e+17,
    "node_network_transmit_packets_total": 5.02e+17,
    "node_nf_conntrack_entries": 2.04e+5,
    "node_network_receive_bytes_total": 3.77e+22,
    "node_network_transmit_bytes_total": 1.44e+22,
}

# THRESHOLD = 0.001  # Mean squared error threshold for anomaly
SEQUENCE_LENGTH = 2
INPUT_DIM = len(FEATURES)
HIDDEN_DIM = 64
NUM_LAYERS = 2
MODEL_PATH = "lstm_model.pth"
FETCH_INTERVAL = 60  # seconds
