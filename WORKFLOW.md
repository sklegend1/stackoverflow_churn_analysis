# راهنمای اجرای پروژه (Project Workflow)

این فایل مراحل اجرای پروژه را برای رویکردهای مختلف توضیح می‌دهد. برای سهولت کار، می‌توانید از اسکریپت خودکار `run_pipeline.py` استفاده کنید.

## روش اجرای سریع (پیشنهادی)

در پوشه اصلی پروژه، دستور زیر را اجرا کنید:

```bash
python run_pipeline.py
```

سپس از منوی نمایش داده شده، گزینه مورد نظر خود را انتخاب کنید (مثلاً گزینه 1 برای اجرای کامل پایپ‌لاین XGBoost).

---

## مراحل اجرای دستی (Manual Execution)

اگر می‌خواهید اسکریپت‌ها را به صورت جداگانه اجرا کنید، ترتیب زیر را رعایت کنید:

### 1. رویکرد XGBoost / Rule-Based

این رویکرد برای مدل‌سازی بر اساس ویژگی‌های استخراج شده (Feature Engineering) است.

1.  **ساخت ماتریس ویژگی‌ها (Feature Matrix):**
    *   اسکریپت: `src/data_engineering/create_feature_matrix.py`
    *   خروجی: `data/processed/feature_matrix.csv`

2.  **ساخت لاگ رویداد هسته (Core Event Log):**
    *   اسکریپت: `src/data_engineering/create_core_eventlog.py`
    *   خروجی: `data/processed/core_eventlog.parquet`

3.  **اجرای مدل (Model Training & Prediction):**
    *   اسکریپت: `src/models/xgboost/tp_ap_xgboost.py`
    *   خروجی: `results/tp_ap_estimates.csv` (پیش‌بینی‌های مدل)

4.  **ارزیابی (Evaluation):**
    *   اسکریپت: `src/evaluation/simple_eval.py`
    *   خروجی: نمایش نتایج دقت (Accuracy, F1-Score) در ترمینال و ذخیره در `results/evaluations/`

---

### 2. رویکرد LSTM (Deep Learning)

این رویکرد برای مدل‌سازی ترتیبی (Sequential) رفتار کاربران است.

1.  **ساخت ماتریس ویژگی‌ها (Feature Matrix):**
    *   اسکریپت: `src/data_engineering/create_feature_matrix.py`

2.  **ساخت لاگ رویداد هسته (Core Event Log):**
    *   اسکریپت: `src/data_engineering/create_core_eventlog.py`

3.  **اجرای مدل LSTM Temporal:**
    *   اسکریپت: `src/models/lstm/lstm_temporal_ap_tp.py`
    *   خروجی: مدل‌های ذخیره شده در `checkpoints/` و نتایج در `results/`

4.  **ارزیابی چند ساله (Multi-Year Evaluation):**
    *   اسکریپت: `src/evaluation/lstm_eval_multi_year.py`

---

## ساختار پوشه‌ها

*   `src/`: کدهای اصلی برنامه
*   `data/`: داده‌های ورودی و خروجی میانی
*   `results/`: نتایج نهایی و نمودارها
*   `docs/`: مستندات پروژه
