# LSTM-Based Real-Time Anomaly Detection System 🚀

این پروژه یک سیستم **تشخیص ناهنجاری بلادرنگ** است که با استفاده از **مدل LSTM** داده‌های سیستمی را پردازش کرده و نتایج را به صورت **متریک Prometheus** اکسپورت می‌کند. این سیستم با داده‌های خروجی Node Exporter و ذخیره شده در **VictoriaMetrics** کار می‌کند.

---

## 📦 ساختار پروژه


---

## ⚙️ توضیح فایل‌ها

### `config.py`
- شامل تنظیمات مانند:
  - URL به VictoriaMetrics
  - ویژگی‌هایی که باید مانیتور شوند
  - Threshold برای تشخیص ناهنجاری
  - ابعاد مدل و پارامترهای آموزشی
  - مسیر مدل ذخیره‌شده
  - فاصله زمانی Fetch داده‌ها

---

### `data_fetcher.py`
- **گرفتن آخرین داده** از VictoriaMetrics با PromQL برای هر feature تعریف شده.
- بازگشت یک آرایه از مقادیر آخر.

---

### `detect_anomalies.py`
- پردازش داده‌های جدید، اعمال پیش‌پردازش و فید دادن به مدل LSTM.
- محاسبه‌ی **خطای MSE**.
- تشخیص آنومالی: اگر خطا از Threshold بیشتر باشد ➔ آنومالی.

---

### `preprocessing.py`
- آماده‌سازی داده‌ها:
  - Scaling مقادیر ورودی با استانداردسازی (mean=0, std=1).
  - تبدیل داده خام به فرم قابل قبول برای LSTM.

---

### `model.py`
- تعریف ساختار مدل LSTM شامل:
  - چندین لایه LSTM
  - لایه Fully Connected برای تولید خروجی
- بارگذاری مدل آموزش داده شده.

---

### `main_prometheus_exporter.py`
- **اصلی‌ترین فایل اجرای پروژه**:
  - اجرا کردن یک سرور Flask روی `/metrics`.
  - جمع‌آوری داده‌های جدید ➔ تشخیص ناهنجاری ➔ آپدیت متریک‌های Prometheus.
  - استفاده از `prometheus_client` برای ثبت متریک‌ها.

---

## 🏗️ روند کلی سیستم

```mermaid
flowchart TD
    A[VictoriaMetrics] -->|PromQL| B[Data Fetcher]
    B --> C[Preprocessing]
    C --> D[LSTM Model]
    D --> E[Compute MSE]
    E --> F[Compare with Threshold]
    F --> G{Is Anomaly?}
    G -->|Yes/No| H[Expose via Flask /metrics]
    H --> I[Prometheus Scrapes Metrics]





---

✅ این شد یک `README.md` **فوق حرفه‌ای و مرتب** که هم برای خودت فوق‌العاده مفیده و هم اگر بخوای پروژه رو در گیت‌هاب آپلود کنی خیلی شیک و کامل به نظر میرسه.

---

دوست داری اگر بخوای فایل `README.md` نهایی رو آماده دانلود کنم؟ 📄  
یا بخوای نسخه‌ی انگلیسی هم کنارش آماده کنم؟ 🔥🎯  
بگی تا برات آماده کنم! 🚀
