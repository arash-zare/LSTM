
# # data_fetcher.py
# import requests
# from config import VICTORIA_METRICS_URL, FEATURES

# def fetch_latest_data():
#     values = []
#     for feature in FEATURES:
#         query = f'{feature}[1m]'  # مقدار 1 دقیقه اخیر
#         try:
#             response = requests.get(VICTORIA_METRICS_URL, params={'query': query})
#             response.raise_for_status()
#             result = response.json().get('data', {}).get('result', [])

#             if result:
#                 # گرفتن آخرین مقدار
#                 value = float(result[0]['values'][-1][1])
#                 print(f"[✔] Fetched {feature}: {value}")
#                 values.append(value)
#             else:
#                 # اگر نتیجه خالی بود
#                 print(f"[⚠️] No data for {feature}, setting 0.0")
#                 values.append(0.0)

#         except Exception as e:
#             print(f"[❌] Error fetching {feature}: {e}")
#             values.append(0.0)

#     return values


# data_fetcher.py

import requests
from config import VICTORIA_METRICS_URL, FEATURES

def fetch_latest_data():
    values = []
    for feature in FEATURES:
        query = f'{feature}[1m]'
        try:
            response = requests.get(VICTORIA_METRICS_URL, params={'query': query})
            response.raise_for_status()
            result = response.json().get('data', {}).get('result', [])

            if result:
                # اگر feature شبکه‌ای بود (که device label داره)
                if "network" in feature:
                    value = None
                    eth0_value = None
                    max_value = 0.0

                    for r in result:
                        device = r.get('metric', {}).get('device', '')
                        last_value = float(r['values'][-1][1])

                        if device == 'eth0':
                            eth0_value = last_value

                        if last_value > max_value:
                            max_value = last_value

                    if eth0_value is not None:
                        value = eth0_value
                        print(f"[✔] Selected eth0 for {feature}: {value}")
                    else:
                        value = max_value
                        print(f"[✔] Selected max device for {feature}: {value}")

                else:
                    # حالت ساده برای featureهای غیر شبکه‌ای
                    value = float(result[0]['values'][-1][1])
                    print(f"[✔] Fetched {feature}: {value}")

                values.append(value)

            else:
                print(f"[⚠️] No data for {feature}, setting 0.0")
                values.append(0.0)

        except Exception as e:
            print(f"[❌] Error fetching {feature}: {e}")
            values.append(0.0)

    return values
