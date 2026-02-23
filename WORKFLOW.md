# راهنمای اجرای پروژه (Project Workflow)

این فایل مراحل اجرای پروژه را برای رویکردهای مختلف توضیح می‌دهد. برای سهولت کار، می‌توانید از اسکریپت خودکار `run_pipeline.py` استفاده کنید.

## پیش‌نیازها

**۱. نصب نیازمندی‌ها:**
قبل از هر کاری، لطفاً کتابخانه‌های مورد نیاز پایتون را نصب کنید:
```bash
pip install -r requirements.txt
```

**۲. داده‌های اولیه:**
قبل از اجرای هر پایپ‌لاین، مطمئن شوید که فایل‌های داده خام پردازش شده‌اند. اگر اولین بار است که پروژه را اجرا می‌کنید، ابتدا **پایپ‌لاین ۳ (Data Engineering)** را اجرا کنید تا `eventlog_v2.parquet` ساخته شود.

---

## ۱. رویکرد XGBoost / Rule-Based

این رویکرد برای مدل‌سازی بر اساس ویژگی‌های استخراج شده (Feature Engineering) است.

1.  **ساخت ماتریس ویژگی‌ها (Feature Matrix):**
    *   **اسکریپت:** `src/data_engineering/create_feature_matrix.py`
    *   **ورودی:** `data/processed/eventlog_v2.parquet`
    *   **خروجی:** `data/processed/feature_matrix.csv`

2.  **ساخت لاگ رویداد هسته (Core Event Log):**
    *   **اسکریپت:** `src/data_engineering/create_core_eventlog.py` (یا `create_core_eventlog_splited.py` برای تولید داده‌های Train/Eval)
    *   **خروجی:** `data/processed/core_eventlog_v1.parquet` (یا `train_v1`/`eval_v1`)
    *   **توجه:** مدل `tp_ap_xgboost.py` به فایل `core_eventlog_train_v1.parquet` نیاز دارد که توسط نسخه `_splited` تولید می‌شود.

3.  **اجرای مدل (Model Training & Prediction):**
    *   **اسکریپت:** `src/models/xgboost/tp_ap_xgboost.py`
    *   **ورودی:** `data/processed/core_eventlog_train_v1.parquet`
    *   **خروجی:** `results/tp_ap_estimates.csv` (پیش‌بینی‌های مدل)

4.  **ارزیابی (Evaluation):**
    *   **اسکریپت:** `src/evaluation/simple_eval.py`
    *   **ورودی:** `results/tp_ap_estimates.csv`
    *   **خروجی:** نتایج ارزیابی در پوشه `results/evaluations`

---

## ۲. رویکرد LSTM (Deep Learning)

این رویکرد برای مدل‌سازی ترتیبی (Sequential) رفتار کاربران است.

1.  **ساخت ماتریس ویژگی‌ها (Feature Matrix):**
    *   **اسکریپت:** `src/data_engineering/create_feature_matrix.py`

2.  **ساخت لاگ رویداد هسته (Core Event Log):**
    *   **اسکریپت:** `src/data_engineering/create_core_eventlog.py` (توصیه: استفاده از نسخه `_splited` برای دیتاست Train)

3.  **اجرای مدل LSTM Temporal:**
    *   **اسکریپت:** `src/models/lstm/lstm_temporal_ap_tp.py`
    *   **ورودی:** `data/processed/core_eventlog_train_v1.parquet`
    *   **خروجی:** مدل‌های ذخیره شده در `checkpoints/` و نتایج در `results/`

4.  **ارزیابی چند ساله (Multi-Year Evaluation):**
    *   **اسکریپت:** `src/evaluation/lstm_eval_multi_year.py`
    *   **ورودی:** نتایج حاصل از مدل LSTM

---

## نکات فنی

*   برای اجرای صحیح اسکریپت‌ها و جلوگیری از مشکلات مسیر فایل (Path Issues)، حتماً از `run_pipeline.py` استفاده کنید یا اسکریپت‌ها را در دایرکتوری خودشان اجرا نمایید.
*   تمام فایل‌های داده در `data/processed` ذخیره می‌شوند.

