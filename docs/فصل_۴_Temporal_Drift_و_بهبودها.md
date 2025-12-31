<div dir="rtl" align="right">

# فصل ۴ — Temporal Drift و بهبودهای WA

---

## ۴.۱ مقدمه

در جلسه‌ی پنجم پروژه، مشاهده‌ی مهمی گزارش شد:

> «پارامترهای استخراج‌شده در بازه‌های زمانی اخیر با دقت بالاتری نسبت به گذشته عمل می‌کنند. نرخ ریزش واقعی در بازه‌ی زمانی 2022 تا 2024 با افزایش بسیار قابل توجهی روبه‌رو است.»

این پدیده **Temporal Drift** یا **Concept Drift** نام دارد. در این فصل:

1. تحلیل علل Drift
2. بهبود فرمول WA (Trend + Recency)
3. تلاش‌های Calibration
4. ارزیابی temporal-aware

---

## ۴.۲ تحلیل Temporal Drift

### ۴.۲.۱ مشاهدات اولیه

با ارزیابی مدل XGBoost روی پرسوناهای مختلف:

| Persona | نرخ Churn واقعی | Precision | Recall | AUC |
|---------|-----------------|-----------|--------|-----|
| 0 | ~۵–۱۷٪ | 0.10–0.32 | 0.75–0.80 | 0.73–0.78 |
| 1 | ~۹–۳۷٪ | 0.12–0.47 | 0.57–0.90 | 0.64–0.68 |
| 2 | ~۲–۲۲٪ | 0.14–0.44 | 0.64–0.96 | 0.77–0.82 |
| 3 | ~۳–۹٪ | 0.06–0.17 | 0.75–1.0 | 0.74–0.84 |
| 4 | ~۷–۳۶٪ | 0.10–0.48 | 0.59–0.89 | 0.67–0.70 |

**منبع:** `tp_ap_xgb_joint_search.csv`

### ۴.۲.۲ علل احتمالی

#### ۱. تأثیر COVID-19 (2020–2021)

دوره‌ی COVID-19 باعث **افزایش موقت فعالیت** شد:
- برنامه‌نویسان بیشتر در خانه بودند
- تقاضا برای آموزش آنلاین افزایش یافت
- بعد از این دوره، بسیاری برنگشتند

#### ۲. ظهور LLM‌ها (2022–2024)

با ظهور ChatGPT و ابزارهای AI:
- نیاز به پرسیدن سؤال کاهش یافت
- پاسخ‌دهندگان کمتر مورد نیاز شدند
- **کاهش ساختاری در مشارکت** رخ داد

#### ۳. تغییرات پلتفرم

Stack Overflow خود تغییراتی در سیاست‌ها داشت که ممکن است روی رفتار کاربران تأثیر گذاشته باشد.

### ۴.۲.۳ تأثیر روی مدل

$$
\text{Rate Ratio} = \frac{\text{Predicted Churn Rate}}{\text{Actual Churn Rate}}
$$

| سال | Rate Ratio |
|-----|------------|
| 2016 | ~1.1 |
| 2018 | ~0.9 |
| 2020 | ~0.7 |
| 2022 | ~0.5 |

مدل با پارامترهای قدیمی، **کمتر از واقع** Churn پیش‌بینی می‌کند.

---

## ۴.۳ بهبود فرمول WA

### ۴.۳.۱ محدودیت WA ساده

$$
WA_{\text{simple}}(u) = \sum_{e \in \text{Events}(u, T_p)} \text{gain}(e)
$$

این فرمول:
- ❌ روند تغییر فعالیت را نمی‌بیند
- ❌ تازگی آخرین فعالیت را نمی‌بیند
- ❌ همه‌ی رویدادها را برابر می‌شمارد

### ۴.۳.۲ فرمول‌های بهبود یافته

#### WA_trend: توجه به روند

پنجره‌ی زمانی به دو نیمه تقسیم می‌شود:

```python
mid_point = window_start + pl.duration(seconds=int(Tp_sec / 2))

features = win.with_columns([
    (pl.col("timestamp") >= mid_point).alias("is_recent"),
])

# recent_gain: مجموع فعالیت نیمه‌ی دوم
# old_gain: مجموع فعالیت نیمه‌ی اول
```

$$
\text{activity\_trend} = \frac{\text{recent\_gain} + 1}{\text{old\_gain} + 1}
$$

$$
WA_{\text{trend}} = WA_{\text{simple}} \times \min(\max(\text{activity\_trend}, 0.5), 2.0)
$$

**تفسیر:**
- اگر فعالیت رو به افزایش است: $WA$ بالاتر می‌رود
- اگر فعالیت رو به کاهش است: $WA$ پایین‌تر می‌آید
- `clip(0.5, 2.0)` از تغییرات افراطی جلوگیری می‌کند

#### WA_recency: توجه به تازگی

$$
\text{recency\_weight} = \exp\left(-\frac{\text{days\_since\_last}}{30}\right)
$$

$$
WA_{\text{recency}} = WA_{\text{simple}} \times \text{recency\_weight}
$$

```python
features = win.with_columns([
    ((cutoff - pl.col("timestamp")).dt.total_seconds() / 86400).alias("days_ago"),
])

recency_weight = ((-pl.col("days_since_last") / 30.0).exp())
```

**تفسیر:**
- اگر آخرین فعالیت دیروز بود: وزن ≈ ۱
- اگر آخرین فعالیت ۳۰ روز پیش بود: وزن ≈ ۰.۳۷
- اگر آخرین فعالیت ۶۰ روز پیش بود: وزن ≈ ۰.۱۴

#### WA_enhanced: ترکیب هر دو

$$
WA_{\text{enhanced}} = WA_{\text{simple}} \times \text{activity\_trend} \times \text{recency\_weight}
$$

```python
WA_enhanced = (
    pl.col("total_gain") *
    pl.col("activity_trend").clip(0.5, 2.0) *
    pl.col("recency_weight")
)
```

### ۴.۳.۳ مقایسه‌ی عملکرد

با استفاده از XGBoost برای بهینه‌سازی $A_p$ روی هر نوع WA:

| WA Type | Median F1 (2016–2022) | بهترین برای |
|---------|----------------------|-------------|
| WA_simple | 0.65 | — |
| WA_trend | 0.67 | کاربران با تغییر رفتار |
| WA_recency | 0.69 | کاربران گاه‌به‌گاه |
| WA_enhanced | 0.68 | ترکیب |

**نتیجه:** `WA_recency` بهترین عملکرد را داشت.

---

## ۴.۴ پیاده‌سازی

### ۴.۴.۱ تابع محاسبه‌ی تمام WA‌ها

```python
def calculate_wa_metrics(win, cutoff, Tp_sec):
    """
    Calculate multiple WA metrics for users in window.
    """
    window_start = cutoff - pl.duration(seconds=int(Tp_sec))
    mid_point = window_start + pl.duration(seconds=int(Tp_sec / 2))

    features = (
        win
        .with_columns([
            (pl.col("timestamp") >= mid_point).alias("is_recent"),
            ((cutoff - pl.col("timestamp")).dt.total_seconds() / 86400).alias("days_ago"),
        ])
        .group_by("user_id")
        .agg([
            pl.col("gain").sum().alias("total_gain"),
            pl.col("gain").filter(pl.col("is_recent")).sum().alias("recent_gain"),
            pl.col("gain").filter(~pl.col("is_recent")).sum().alias("old_gain"),
            pl.col("days_ago").min().alias("days_since_last"),
        ])
    )

    features = features.with_columns([
        # WA_simple
        pl.col("total_gain").alias("WA_simple"),
        
        # Activity trend
        ((pl.col("recent_gain") + 1.0) / (pl.col("old_gain") + 1.0))
            .alias("activity_trend"),
        
        # Recency weight
        ((-pl.col("days_since_last") / 30.0).exp())
            .alias("recency_weight"),
    ])

    # Apply combinations
    features = features.with_columns([
        (pl.col("total_gain") * pl.col("activity_trend").clip(0.5, 2.0))
            .alias("WA_trend"),
        (pl.col("total_gain") * pl.col("recency_weight"))
            .alias("WA_recency"),
        (pl.col("total_gain") * pl.col("activity_trend").clip(0.5, 2.0) * pl.col("recency_weight"))
            .alias("WA_enhanced"),
    ])

    return features
```

### ۴.۴.۲ بهینه‌سازی $A_p$ با XGBoost

```python
for wa_type in ['WA_simple', 'WA_trend', 'WA_recency', 'WA_enhanced']:
    # Train XGBoost to find optimal Ap
    X = wa_metrics[[wa_type]].to_pandas()
    y = churn_labels
    
    model = XGBClassifier(...)
    model.fit(X, y)
    
    # Find best threshold using precision-recall curve
    probs = model.predict_proba(X)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_threshold = thresholds[np.argmax(f1_scores)]
```

---

## ۴.۵ تلاش‌های Calibration

### ۴.۵.۱ ایده‌ی Calibration

اگر بتوانیم $A_p$ را بر اساس سال تنظیم کنیم، شاید بتوانیم Drift را جبران کنیم:

$$
A_p^{(\text{year})} = A_p^{(\text{base})} \times \text{multiplier}(\text{year})
$$

### ۴.۵.۲ روش Ratio Scaling

```python
# محاسبه‌ی نرخ واقعی و پیش‌بینی شده
actual_rate = y_true.mean()
pred_rate = y_pred.mean()

# تنظیم Ap
multiplier = pred_rate / actual_rate  # یا بالعکس
Ap_calibrated = Ap_original * multiplier
```

### ۴.۵.۳ روش Quantile Matching

```python
# تطبیق توزیع WA با توزیع Churn
actual_churn_quantile = np.quantile(WA[y_true == 1], 0.5)
Ap_calibrated = actual_churn_quantile
```

### ۴.۵.۴ نتایج Calibration

| روش | Rate Ratio بعد | F1 بعد |
|-----|---------------|--------|
| Ratio Scaling | ~0.95 | 0.58 |
| Quantile Matching | ~1.1 | 0.55 |
| Temperature Scaling | ~0.85 | 0.60 |

**مشکل:** بهبود Rate Ratio معمولاً باعث کاهش F1 شد!

### ۴.۵.۵ علت شکست Calibration

> «تغییرات Temporal یک‌نواخت نیستند. برخی پرسوناها بیشتر تحت تأثیر قرار گرفته‌اند. یک ضریب سراسری کافی نیست.»

---

## ۴.۶ ارزیابی Temporal-aware

### ۴.۶.۱ استفاده از Year Multipliers

```python
year_multipliers = pd.read_csv("temporal_year_multipliers.csv")

# تنظیم Ap بر اساس سال
year_multiplier = year_multipliers.get(CUTOFF_YEAR, 1.0)
Ap_map = {persona: Ap * year_multiplier for persona, Ap in Ap_map.items()}
```

### ۴.۶.۲ اسکریپت ارزیابی

```python
# simple_eval_temporal.py

WA_METHOD = 'WA_recency'  # بهترین روش
CUTOFF_YEAR = 2016
GRACE_DAYS = 180

# بارگذاری پارامترها
best_params = pd.read_csv(f"tp_ap_xgb_enhanced_best_params_{WA_METHOD}.csv")

# بارگذاری ضرایب سالانه
year_multipliers = pd.read_csv("temporal_year_multipliers.csv")
year_multiplier = year_multipliers.get(CUTOFF_YEAR, 1.0)

# تنظیم Ap
Ap_map = {p: Ap * year_multiplier for p, Ap in Ap_map.items()}
```

### ۴.۶.۳ نتایج Enhanced WA

مقایسه‌ی روش‌های مختلف WA (از `enhanced_wa_best_params.csv`):

| Persona | WA Type | Median F1 | Median Precision | Median Recall | AUC |
|---------|---------|-----------|------------------|---------------|-----|
| 0 | WA_trend | 0.53 | 0.42 | 0.76 | 0.72 |
| 1 | WA_trend | 0.60 | 0.47 | 0.84 | 0.67 |
| 2 | WA_trend | 0.62 | 0.50 | 0.79 | 0.71 |
| 3 | WA_recency | 0.56 | 0.44 | 0.77 | 0.76 |
| 4 | WA_recency | 0.60 | 0.47 | 0.88 | 0.69 |
| 5 | WA_recency | 0.60 | 0.47 | 0.85 | 0.68 |

**نتیجه:** `WA_trend` و `WA_recency` عملکرد بهتری از `WA_simple` دارند.

---

## ۴.۷ محدودیت‌های باقی‌مانده

### ۴.۷.۱ پارامترهای Static

حتی با Calibration، پارامترها در طول زمان ثابت هستند. مدل نمی‌تواند:
- تغییرات آینده را پیش‌بینی کند
- به رویدادهای جدید سازگار شود

### ۴.۷.۲ نیاز به مدل Sequence

> «داده‌ها ماهیت توالی زمانی دارند. رفتار کاربر به مرور شکل می‌گیرد و یک snapshot آخر، رفتار را درست نشان نمی‌دهد.»

این محدودیت منجر به توسعه‌ی مدل LSTM در فصل بعد شد.

---

## ۴.۸ فایل‌های مرتبط

| فایل | نقش |
|------|-----|
| `tp_ap_xgboost_enhanced.py` | بهینه‌سازی با ۴ نوع WA |
| `simple_eval_temporal.py` | ارزیابی با تنظیم سالانه |
| `calibrate_ap_combined.py` | تلاش‌های Calibration |
| `calibrate_ap_advanced.py` | روش‌های پیشرفته Calibration |
| `temporal_year_multipliers.csv` | ضرایب تنظیم سالانه |

---

## ۴.۹ خلاصه‌ی فصل

در این فصل:

- ✅ پدیده‌ی Temporal Drift تحلیل شد
- ✅ علل Drift شناسایی شد: COVID، LLM، تغییرات پلتفرم
- ✅ فرمول‌های بهبود یافته‌ی WA توسعه داده شد:
  - `WA_trend`: توجه به روند
  - `WA_recency`: توجه به تازگی
  - `WA_enhanced`: ترکیب
- ✅ `WA_recency` بهترین عملکرد را داشت
- ⚠️ Calibration موفقیت کامل نداشت
- ⚠️ محدودیت‌های ذاتی Rule-based شناسایی شد

**نتیجه‌گیری:** برای غلبه بر محدودیت‌های ذاتی سیستم Rule-based، نیاز به مدل Sequence-based (LSTM) است.

---

**پایان فصل ۴**

</div>
