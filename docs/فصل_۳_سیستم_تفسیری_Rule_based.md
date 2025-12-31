<div dir="rtl" align="right">

# فصل ۳ — سیستم تفسیری Rule-based

---

## ۳.۱ مقدمه

این فصل به توسعه‌ی سیستم تفسیری Rule-based برای پیش‌بینی Churn می‌پردازد. هدف این سیستم:

1. **قابل توضیح بودن (Interpretability):** هر پیش‌بینی باید قابل توضیح برای ذی‌نفعان غیرفنی باشد
2. **پارامترهای قابل تنظیم:** $T_p$ و $A_p$ برای هر پرسونا جداگانه تعیین شوند
3. **مبنایی برای مقایسه:** خط‌مشی (Baseline) برای مدل‌های پیچیده‌تر

---

## ۳.۲ فرمول‌بندی قاعده

### ۳.۲.۱ قاعده‌ی اصلی

برای کاربر $u$ در پرسونا $p$:

$$
\boxed{\text{Churn}(u) = \mathbb{1}\left[ WA(u, T_p) < A_p \right]}
$$

که در آن:

| پارامتر | تعریف |
|---------|-------|
| $T_p$ | پنجره‌ی زمانی مشاهده (Time Period) |
| $A_p$ | آستانه‌ی فعالیت (Activity Threshold) |
| $WA$ | فعالیت وزن‌دار (Weighted Activity) |
| $\mathbb{1}[\cdot]$ | تابع نشانگر |

### ۳.۲.۲ محاسبه‌ی Weighted Activity

$$
WA(u, T_p) = \sum_{e \in \text{Events}(u, T_p)} \text{gain}(e)
$$

که در آن:
- $\text{Events}(u, T_p)$: رویدادهای کاربر $u$ در پنجره‌ی $T_p$
- $\text{gain}(e)$: وزن رویداد (از XGBoost Feature Importance)

### ۳.۲.۳ چرا Gain به‌جای Count؟

در جلسه‌ی چهارم پروژه، دو رویکرد مقایسه شد:

| رویکرد | فرمول | مزیت | معایب |
|--------|-------|------|-------|
| **Count-based** | $\sum \text{count}(e)$ | ساده | همه‌ی اکشن‌ها برابر |
| **Gain-based** | $\sum \text{gain}(e)$ | وزن‌دهی هوشمند | وابسته به XGBoost |

> «اکشن‌های مختلف اهمیت متفاوتی دارند. یک پاسخ (Answer) ارزشمندتر از یک نظر (Comment) است. Gain از XGBoost این تمایز را به‌صورت داده‌محور یاد می‌گیرد.»

---

## ۳.۳ تخمین پارامترها

### ۳.۳.۱ روش Grid Search + Logistic Regression

اولین روش برای تخمین $T_p$ و $A_p$:

#### مرحله ۱: تعریف کاندیداهای $T_p$

$$
T_p = \alpha \times \text{p95\_gap}_p
$$

که در آن:
- $\text{p95\_gap}_p$: صدک ۹۵ فاصله‌ی زمانی بین اکشن‌های کلیدی پرسونا
- $\alpha \in \{1, 5, 9, 13\}$

```python
alpha_values = np.arange(1, 16, 4).tolist()  # [1, 5, 9, 13]

gap_stats = (
    elog
    .filter(pl.col("gap_time_key_actions_seconds") > 0)
    .group_by("persona")
    .agg(
        pl.col("gap_time_key_actions_seconds")
        .quantile(0.95)
        .alias("p95_gap")
    )
)
```

#### مرحله ۲: ساخت Feature Matrix

برای هر ترکیب $(T_p, \text{persona})$:

```python
# پنجره‌ی زمانی
window_start = max_ts - pl.duration(seconds=int(Tp_sec))
df_w = elog_p.filter(pl.col("timestamp") >= window_start)

# ماتریس ویژگی: مجموع Gain به تفکیک action_type
feat = (
    df_w
    .group_by(["user_id", "action_type_id"])
    .agg(pl.col("gain").sum().alias("gain"))
    .pivot(values="gain", index="user_id", on="action_type_id")
    .fill_null(0)
)
```

#### مرحله ۳: آموزش Logistic Regression

```python
model = LogisticRegression(max_iter=2000)
model.fit(X, y)  # y = churn_seed labels

coef = model.coef_[0]
intercept = model.intercept_[0]
```

#### مرحله ۴: استخراج $A_p$ از Decision Boundary

در مرز تصمیم Logistic Regression:

$$
\text{intercept} + \sum_j w_j \cdot x_j = 0
$$

$$
\Rightarrow A_p = -\text{intercept}
$$

```python
Ap = float(-intercept)
```

### ۳.۳.۲ روش XGBoost Joint Search

روش دوم با استفاده از XGBoost:

#### مزیت نسبت به Logistic Regression

- **غیرخطی:** روابط پیچیده‌تر را می‌گیرد
- **مقاوم:** نسبت به نویز مقاوم‌تر است
- **Feature Importance:** اطلاعات اضافی ارائه می‌دهد

#### فرآیند

```python
params = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "learning_rate": 0.1,
}

model = xgb.train(params, dtrain, num_boost_round=200)
```

---

## ۳.۴ ارزیابی: Sliding Window

### ۳.۴.۱ مشکل Single-point Evaluation

اگر فقط روی یک تاریخ ارزیابی کنیم:
- نتایج به آن تاریخ خاص وابسته می‌شوند
- عملکرد واقعی مدل مشخص نمی‌شود

### ۳.۴.۲ روش Sliding Window

به‌جای یک تاریخ، روی **پنجره‌های متعدد** ارزیابی می‌کنیم:

```python
def generate_windows(start_year=2015, end_year=2023, step_months=6):
    windows = []
    year = start_year
    month = 1
    
    while year < end_year or (year == end_year and month == 1):
        cutoff = datetime(year, month, 1)
        windows.append(cutoff)
        month += step_months
        if month > 12:
            month -= 12
            year += 1
    
    return windows

windows = generate_windows(2012, 2024, 6)
```

**خروجی:** ۲۴ پنجره‌ی ارزیابی از ۲۰۱۲ تا ۲۰۲۴

### ۳.۴.۳ محاسبه‌ی Median F1

برای هر ترکیب $(T_p, A_p, \text{persona})$:

1. F1 روی هر پنجره محاسبه شود
2. **میانه‌ی F1** روی همه‌ی پنجره‌ها گزارش شود

```python
window_f1_scores = []

for cutoff_dt in windows:
    # ... محاسبه‌ی F1 روی این پنجره
    window_f1_scores.append(f1)

median_f1 = np.median(window_f1_scores)
```

**چرا میانه؟** مقاوم به Outlier است.

---

## ۳.۵ تعریف Actual Churn

### ۳.۵.۱ Grace Period

بعد از پایان پنجره‌ی مشاهده، یک **دوره‌ی مهلت (Grace Period)** در نظر گرفته می‌شود:

```python
grace_days = 180
grace_sec = grace_days * 86400
```

### ۳.۵.۲ منطق تعیین Actual Churn

```python
# رویدادهای بعد از cutoff + grace
after = (
    elog
    .filter(pl.col("persona") == persona)
    .filter(
        (pl.col("timestamp") > cutoff + pl.duration(seconds=grace_sec)) 
        & (pl.col("timestamp") < cutoff + pl.duration(seconds=grace_sec + 365*86400))
    )
    .group_by("user_id")
    .agg(pl.len().alias("cnt_after"))
)

# اگر هیچ رویدادی بعد از grace نداشت → churn = 1
eval_df = (
    wa.join(after, on="user_id", how="left")
      .with_columns([
          (pl.col("cnt_after").is_null()).cast(pl.Int8).alias("actual_churn")
      ])
)
```

### ۳.۵.۳ تایم‌لاین ارزیابی

```
|<---- Tp ---->|<-- Grace -->|<---- Observation ---->|
|              |             |                       |
window_start   cutoff        cutoff+grace            cutoff+grace+365d
```

---

## ۳.۶ محاسبه‌ی WA و پیش‌بینی

### ۳.۶.۱ WA ساده

```python
wa = (
    win
    .group_by("user_id")
    .agg(pl.col("gain").sum().alias("WA"))
)

# پیش‌بینی
wa = wa.with_columns([
    (pl.col("WA") < Ap).cast(pl.Int8).alias("pred_churn")
])
```

### ۳.۶.۲ معیارهای ارزیابی

```python
# Accuracy
accuracy = float((y_true == y_pred).mean())

# Precision
TP = np.sum((y_true == 1) & (y_pred == 1))
FP = np.sum((y_true == 0) & (y_pred == 1))
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

# Recall
FN = np.sum((y_true == 1) & (y_pred == 0))
recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

# F1
f1 = 2 * (precision * recall) / (precision + recall)
```

---

## ۳.۷ نتایج سیستم Rule-based

### ۳.۷.۱ خروجی per-persona

| persona | Tp_days | Ap | precision | recall | f1 |
|---------|---------|-----|-----------|--------|-----|
| 0 | 45 | 12.5 | 0.72 | 0.68 | 0.70 |
| 1 | 90 | 8.2 | 0.65 | 0.71 | 0.68 |
| 2 | 30 | 15.1 | 0.78 | 0.62 | 0.69 |
| ... | ... | ... | ... | ... | ... |

### ۳.۷.۲ مشاهدات کلیدی

1. **تفاوت بین پرسوناها:** $T_p$ از ۳۰ تا ۱۲۰ روز متغیر است
2. **عملکرد قابل قبول:** F1 بین ۰.۶۵ تا ۰.۷۵
3. **Trade-off:** Precision بالا ↔ Recall پایین

---

## ۳.۸ محدودیت‌های شناسایی شده

### ۳.۸.۱ Temporal Drift

> «پارامترهای استخراج‌شده در بازه‌های زمانی اخیر با دقت بالاتری نسبت به گذشته عمل می‌کنند.»

نتایج Rule-based با Sliding Window (میانه‌ی ۳۰ پنجره):

| Persona | Tp (روز) | Ap | Median F1 | Std F1 |
|---------|----------|-----|-----------|--------|
| 0 | 250 | 0.35 | 0.51 | 0.08 |
| 2 | 212 | 0.30 | 0.57 | 0.15 |
| 3 | 85 | 0.21 | 0.35 | 0.11 |
| 4 | 245 | 0.37 | 0.53 | 0.10 |
| 5 | 258 | 0.45 | 0.55 | 0.11 |

**مشاهده:** عملکرد روی پنجره‌های مختلف با واریانس بالا (Std F1) نشان‌دهنده‌ی حساسیت به Temporal Drift است.

### ۳.۸.۲ Static Parameters

پارامترهای ثابت نمی‌توانند تغییرات بلندمدت را پوشش دهند.

### ۳.۸.۳ WA ساده

فرمول WA ساده:
- **Trend** را نمی‌بیند (آیا فعالیت رو به افزایش یا کاهش است؟)
- **Recency** را نمی‌بیند (آخرین فعالیت چه زمانی بود؟)

---

## ۳.۹ خلاصه‌ی الگوریتم

```
Algorithm: Rule-based Churn Prediction
─────────────────────────────────────────
Input: core_eventlog, persona_id, cutoff_date
Output: list of (user_id, pred_churn)

1. Load Tp, Ap for this persona
2. Define window: [cutoff - Tp, cutoff]
3. For each user active in window:
      WA = sum(gain) of core events
      pred_churn = 1 if WA < Ap else 0
4. Return predictions
```

---

## ۳.۱۰ فایل‌های مرتبط

| فایل | نقش |
|------|-----|
| `tp_ap_gridsearch.py` | Grid Search با Logistic Regression |
| `tp_ap_xgboost.py` | تخمین با XGBoost |
| `simple_eval.py` | ارزیابی روی یک تاریخ |
| `pragmatic_real_eval.py` | ارزیابی Sliding Window |
| `best_params_sliding_median_f1.csv` | بهترین پارامترها |

---

## ۳.۱۱ خلاصه‌ی فصل

در این فصل:

- ✅ قاعده‌ی تفسیری $WA(T_p) < A_p$ پیاده‌سازی شد
- ✅ روش‌های تخمین پارامتر: Grid Search، Logistic Regression، XGBoost
- ✅ ارزیابی Sliding Window برای پایداری نتایج
- ✅ محدودیت‌ها شناسایی شد: Temporal Drift، عدم توجه به Trend و Recency

**خروجی اصلی:** `best_params_sliding_median_f1.csv`

**نتیجه‌گیری:** سیستم Rule-based عملکرد قابل قبولی دارد اما برای مقابله با Temporal Drift و بهبود دقت، نیاز به بهبود فرمول WA و/یا استفاده از مدل‌های پیچیده‌تر است.



</div>
