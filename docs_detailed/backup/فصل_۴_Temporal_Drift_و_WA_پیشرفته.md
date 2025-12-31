<div dir="rtl" align="right">

# فصل ۴ — Temporal Drift و WA‌های پیشرفته
## نسخه‌ی تکنیکال با جزئیات کامل

---

## ۴.۱ مقدمه

### ۴.۱.۱ مشکل کشف‌شده

در فصل قبل دیدیم که سیستم Rule-based با فرمول ساده‌ی $WA < A_p$ کار می‌کند. اما این روش محدودیت‌هایی دارد:

| محدودیت | توضیح ساده | مثال |
|---------|------------|------|
| **Static Parameters** | پارامترها ثابت‌اند | آستانه‌ی ۲۰۱۲ با ۲۰۲۳ فرق دارد |
| **Simple WA** | فقط مجموع را می‌بیند | فرقی بین ۵+۵ و ۱۰+۰ نمی‌گذارد |
| **No Trend** | روند را نمی‌بیند | نمی‌فهمد فعالیت رو به کاهش است یا افزایش |
| **No Recency** | زمان مهم نیست | فعالیت دیروز با فعالیت ۲ ماه پیش برابر است |

**سؤال کلیدی:** چگونه این مشکلات را حل کنیم بدون اینکه به مدل پیچیده (LSTM) نیاز داشته باشیم؟

### ۴.۱.۲ Temporal Drift چیست؟

**به زبان ساده:** Temporal Drift یعنی «قواعد بازی عوض شده». چیزی که در گذشته نشانه‌ی Churn بود، الان شاید نباشد.

**تعریف ریاضی:**

$$
P(Y|X)_{t_1} \neq P(Y|X)_{t_2}
$$

| نماد | معنی |
|------|------|
| $P(Y,X)$ | احتمال Churn با توجه به ویژگی‌ها |
| $t_1, t_2$ | دو زمان مختلف (مثلاً ۲۰۱۲ و ۲۰۲۳) |

> **💡 مثال مفهومی:**
> 
> رفتار کاربران Stack Overflow در سال ۲۰۱۲ (دوران طلایی) با سال ۲۰۲۳ (رقابت با ChatGPT) متفاوت است:
> 
> - **۲۰۱۲:** کاربران بیشتر سؤال می‌پرسیدند، پاسخ‌دهی فعال‌تر بود، میانگین WA بالا
> - **۲۰۲۳:** کاربران ابتدا ChatGPT را امتحان می‌کنند، سؤالات پیچیده‌تر شده، میانگین WA کمتر
> 
> یک مدل آموزش‌دیده روی ۲۰۱۲ روی ۲۰۲۳ خوب کار نمی‌کند — این Temporal Drift است!

### ۴.۱.۳ مثال عددی Temporal Drift

```
پرسونای Power User:
──────────────────────────────────────────────────────────
سال 2012:
  - میانگین WA = 2.5
  - Churn Rate = 15%
  - آستانه‌ی بهینه Ap = 0.8

سال 2022:
  - میانگین WA = 1.8  ← کاهش ۲۸٪
  - Churn Rate = 25%  ← افزایش ۶۷٪
  - آستانه‌ی بهینه Ap = 0.5  ← نیاز به تنظیم مجدد!
──────────────────────────────────────────────────────────
```

---

## ۴.۲ Enhanced WA: چهار نسخه

برای مقابله با محدودیت‌های WA ساده، چهار نسخه‌ی پیشرفته طراحی شد:

### ۴.۲.۱ WA_simple (Baseline)

**ایده:** مجموع ساده‌ی امتیازات در پنجره‌ی زمانی

$$
WA_{simple} = \sum_{e \in window} \text{gain}(e)
$$

**مشکل:** فرقی بین فعالیت اخیر و قدیمی نمی‌گذارد.

### ۴.۲.۲ WA_trend

**ایده:** کاربری که فعالیتش **رو به کاهش** است، احتمال Churn بیشتری دارد.

**روش محاسبه (از کد واقعی):**

۱. پنجره را به دو نیمه تقسیم کن:
   - نیمه‌ی اول (old): از `window_start` تا `mid_point`
   - نیمه‌ی دوم (recent): از `mid_point` تا `cutoff`

۲. محاسبه‌ی activity_trend:

$$
activity\_trend = \frac{\sum_{recent} gain + 1}{\sum_{old} gain + 1}
$$

۳. محاسبه‌ی WA_trend:

$$
WA_{trend} = WA_{simple} \times \text{clip}(activity\_trend, 0.5, 2.0)
$$

**تفسیر:**

| مقدار activity_trend | معنی | ضریب |
|---------------------|------|------|
| $> 1$ | فعالیت اخیر بیشتر از قبل | تقویت WA |
| $= 1$ | فعالیت ثابت | بدون تغییر |
| $< 1$ | فعالیت اخیر کمتر از قبل | کاهش WA |

> **💡 مثال:**
> 
> | کاربر | نیمه اول | نیمه دوم | activity_trend | تفسیر |
> |-------|---------|---------|----------------|--------|
> | **A** | gain=0.8 | gain=0.2 | (0.2+1)/(0.8+1) = 0.67 | رو به کاهش |
> | **B** | gain=0.2 | gain=0.8 | (0.8+1)/(0.2+1) = 1.5 | رو به افزایش |

### ۴.۲.۳ WA_recency

**ایده:** فعالیت **اخیر** مهم‌تر از فعالیت قدیمی است.

**روش محاسبه (از کد واقعی):**

۱. محاسبه‌ی فاصله از آخرین فعالیت:

$$
days\_since\_last = \min_{e \in window}(\text{cutoff} - \text{timestamp}(e))
$$

۲. محاسبه‌ی recency_weight با $\tau = 30$ روز:

$$
recency\_weight = \exp\left(-\frac{days\_since\_last}{30}\right)
$$

۳. محاسبه‌ی WA_recency:

$$
WA_{recency} = WA_{simple} \times recency\_weight
$$

**تفسیر:**

| days_since_last | recency_weight | معنی |
|-----------------|----------------|------|
| 0 روز | 1.00 | همین الان فعال بوده |
| 30 روز | 0.37 | یک ماه پیش |
| 60 روز | 0.14 | دو ماه پیش |
| 90 روز | 0.05 | سه ماه پیش |

> **💡 مثال:**
> 
> دو کاربر با WA_simple = 1.0:
> 
> | کاربر | آخرین فعالیت | recency_weight | WA_recency |
> |-------|-------------|----------------|------------|
> | **A** | ۵ روز پیش | 0.85 | 0.85 |
> | **B** | ۶۰ روز پیش | 0.14 | 0.14 |
> 
> کاربر B با وجود WA_simple یکسان، ریسک Churn بالاتری دارد.

### ۴.۲.۴ WA_enhanced

**ایده:** ترکیب trend و recency

$$
WA_{enhanced} = WA_{simple} \times \text{clip}(activity\_trend, 0.5, 2.0) \times recency\_weight
$$

**مزیت:** هم روند و هم زمان آخرین فعالیت را در نظر می‌گیرد.

### ۴.۲.۵ WA_consistent (اختیاری)

**ایده:** کاربران با فعالیت **منظم** کمتر Churn می‌کنند.

$$
gap\_cv = \frac{\text{std}(gaps)}{\text{mean}(gaps) + 1}
$$

$$
WA_{consistent} = WA_{simple} \times \frac{2}{1 + gap\_cv}
$$

---

## ۴.۳ مقایسه‌ی چهار نسخه‌ی WA

### ۴.۳.۱ مثال مقایسه‌ای

دو کاربر با WA_simple یکسان:


کاربر A: (رو به کاهش، غیرفعال اخیراً)
─────────────────────────────────────
نیمه اول: gain = 0.8
نیمه دوم: gain = 0.2
آخرین فعالیت: ۴۵ روز پیش

WA_simple = 1.0
activity_trend = (0.2+1)/(0.8+1) = 0.67 → clip = 0.67
recency_weight = exp(-45/30) = 0.22

WA_trend = 1.0 × 0.67 = 0.67
WA_recency = 1.0 × 0.22 = 0.22
WA_enhanced = 1.0 × 0.67 × 0.22 = 0.15

کاربر B: (رو به افزایش، فعال اخیراً)
─────────────────────────────────────
نیمه اول: gain = 0.2
نیمه دوم: gain = 0.8
آخرین فعالیت: ۵ روز پیش

WA_simple = 1.0
activity_trend = (0.8+1)/(0.2+1) = 1.5
recency_weight = exp(-5/30) = 0.85

WA_trend = 1.0 × 1.5 = 1.5
WA_recency = 1.0 × 0.85 = 0.85
WA_enhanced = 1.0 × 1.5 × 0.85 = 1.27


**نتیجه:**

| WA Type | کاربر A | کاربر B | تفاوت |
|---------|---------|---------|-------|
| WA_simple | 1.0 | 1.0 | ❌ صفر |
| WA_trend | 0.67 | 1.5 | ✅ 2.2× |
| WA_recency | 0.22 | 0.85 | ✅ 3.9× |
| WA_enhanced | 0.15 | 1.27 | ✅ **8.5×** |

**نتیجه‌گیری:** WA_enhanced بهترین تفکیک را ایجاد می‌کند.
WA_trend = +0.3 (مثبت)
WA_recency = 0.9 (زیاد، چون اخیراً قوی)


**نتیجه:**

| WA Type | کاربر A | کاربر B | تفاوت؟ |
|---------|---------|---------|--------|
| WA_simple | 1.5 | 1.5 | ❌ نه |
| WA_trend | -0.3 | +0.3 | ✅ بله |
| WA_recency | 0.4 | 0.9 | ✅ بله |

### ۴.۳.۲ قدرت تشخیص هر نوع

| WA Type | چه چیزی می‌بیند | چه چیزی نمی‌بیند |
|---------|----------------|-----------------|
| WA_simple | کل فعالیت | زمان، روند |
| WA_trend | روند فعالیت (افزایشی/کاهشی) | زمان آخرین فعالیت |
| WA_recency | زمان آخرین فعالیت | روند تغییرات |
| WA_enhanced | هم روند، هم زمان | — |

> **💡 تفسیر عملی:**
> 
> چهار نوع WA درباره‌ی یک کاربر SO چه می‌گویند:
> 
> - **WA_simple:** «این کاربر مجموعاً ۸ Answer نوشته.»
> - **WA_trend:** «فعالیت نیمه‌ی دوم ۰.۶۷ برابر نیمه‌ی اول است (رو به کاهش).»
> - **WA_recency:** «آخرین فعالیتش ۴۵ روز پیش بوده (وزن = ۰.۲۲).»
> - **WA_enhanced:** «هم رو به کاهش است، هم مدتی غیرفعال — ریسک بالا!»

---

## ۴.۴ پیاده‌سازی کامل

> **منبع کد:** `enhanced_interpretable_churn_rule.py`

### ۴.۴.۱ تابع اصلی محاسبه‌ی Enhanced WA

این کد **واقعی** از پروژه است که همه‌ی انواع WA را محاسبه می‌کند:

```python
# File: enhanced_interpretable_churn_rule.py

def calculate_enhanced_wa(elog_window, cutoff, Tp_sec):
    """
    Calculate Enhanced Weighted Activity with multiple signals:

    1. WA_simple: sum(gain) - baseline
    2. WA_enhanced: combines trend + recency + consistency

    Returns: DataFrame with user_id and various WA metrics
    """
    window_start = cutoff - pl.duration(seconds=int(Tp_sec))

    # Get events in window
    win = elog_window.filter(
        (pl.col("timestamp") >= window_start) &
        (pl.col("timestamp") <= cutoff)
    )

    if win.height == 0:
        return None

    # Calculate temporal midpoint for trend analysis
    mid_point = window_start + pl.duration(seconds=int(Tp_sec / 2))

    # Extract features per user
    features = (
        win
        .with_columns([
            (pl.col("timestamp") >= mid_point).alias("is_recent"),
            ((cutoff - pl.col("timestamp")).dt.total_seconds() / 86400).alias("days_ago"),
        ])
        .group_by("user_id")
        .agg([
            # Basic metrics
            pl.col("gain").sum().alias("total_gain"),
            pl.len().alias("event_count"),

            # Temporal split for trend detection
            pl.col("gain").filter(pl.col("is_recent")).sum().alias("recent_gain"),
            pl.col("gain").filter(~pl.col("is_recent")).sum().alias("old_gain"),

            # Recency
            pl.col("days_ago").min().alias("days_since_last_activity"),

            # Consistency (inter-event time gaps)
            pl.col("timestamp").sort().diff().dt.total_seconds().drop_nulls().std().alias("gap_std"),
            pl.col("timestamp").sort().diff().dt.total_seconds().drop_nulls().mean().alias("gap_mean"),
        ])
    )

    # Calculate derived metrics
    features = features.with_columns([
        # Activity trend: ratio of recent to old activity
        ((pl.col("recent_gain") + 1.0) / (pl.col("old_gain") + 1.0)).alias("activity_trend"),

        # Recency weight: exponential decay (λ = 1/30 days)
        ((-pl.col("days_since_last_activity") / 30.0).exp()).alias("recency_weight"),

        # Consistency: coefficient of variation
        (pl.col("gap_std") / (pl.col("gap_mean") + 1.0)).alias("gap_cv"),
    ])

    # Create all WA variants
    features = features.with_columns([
        # WA_simple: traditional approach
        pl.col("total_gain").alias("WA_simple"),

        # WA_trend: incorporates activity trend
        (
            pl.col("total_gain") *
            pl.col("activity_trend").clip(0.5, 2.0)
        ).alias("WA_trend"),

        # WA_recency: incorporates recency
        (
            pl.col("total_gain") *
            pl.col("recency_weight")
        ).alias("WA_recency"),

        # WA_enhanced: combines trend + recency
        (
            pl.col("total_gain") *
            pl.col("activity_trend").clip(0.5, 2.0) *
            pl.col("recency_weight")
        ).alias("WA_enhanced"),

        # WA_consistent: bonus for regular users
        (
            pl.col("total_gain") *
            (2.0 / (1.0 + pl.col("gap_cv")))
        ).alias("WA_consistent"),
    ])

    return features
```

**نکات کلیدی کد:**

| بخش | توضیح |
|-----|-------|
| `mid_point` | نقطه‌ی میانی پنجره برای تشخیص trend |
| `is_recent` | فعالیت در نیمه‌ی دوم پنجره |
| `activity_trend` | نسبت فعالیت اخیر به قدیم |
| `recency_weight` | وزن نمایی با $\tau = 30$ روز |
| `clip(0.5, 2.0)` | محدود کردن ضریب trend |

### ۴.۴.۲ فرمول‌های ریاضی

از کد بالا، فرمول‌های دقیق:

$$
WA_{simple} = \sum_{e \in window} gain(e)
$$

$$
activity\_trend = \frac{\sum_{recent} gain + 1}{\sum_{old} gain + 1}
$$

$$
recency\_weight = \exp\left(-\frac{days\_since\_last}{30}\right)
$$

$$
WA_{trend} = WA_{simple} \times \text{clip}(activity\_trend, 0.5, 2.0)
$$

$$
WA_{recency} = WA_{simple} \times recency\_weight
$$

$$
WA_{enhanced} = WA_{simple} \times \text{clip}(activity\_trend, 0.5, 2.0) \times recency\_weight
$$

---

## ۴.۵ Grid Search و ارزیابی

> **منبع کد:** `enhanced_interpretable_churn_rule.py`

### ۴.۵.۱ تنظیمات Grid Search

```python
# File: enhanced_interpretable_churn_rule.py

# Parameters
GRACE_DAYS = 180
GRACE_SEC = GRACE_DAYS * 86400
CHURN_CHECK_WINDOW_DAYS = 365
CHURN_CHECK_WINDOW_SEC = CHURN_CHECK_WINDOW_DAYS * 86400

# Tp range to test (in days)
TP_DAYS_RANGE = [60, 90, 120, 150, 180, 210, 240, 270]

# Beta for F-beta score
BETA = 1.0  # F1 score

# Cutoff dates for temporal cross-validation
TRAIN_CUTOFFS = [
    datetime(2016, 1, 1),
    datetime(2017, 1, 1),
    datetime(2018, 1, 1),
    datetime(2019, 1, 1),
    datetime(2020, 1, 1),
    datetime(2021, 1, 1),
    datetime(2022, 1, 1),
    datetime(2023, 1, 1),
]

# Ap candidates (percentiles of WA distribution)
AP_PERCENTILES = np.arange(10, 91, 5)  # 10th, 15th, ..., 90th percentile
```

### ۴.۵.۲ تابع ارزیابی

```python
# File: enhanced_interpretable_churn_rule.py

def evaluate_ap_tp(elog_persona, persona, cutoff, Tp_sec):
    """
    For given persona, cutoff, and Tp:
    1. Calculate various WA metrics
    2. Get actual churn labels
    3. Test multiple Ap values (percentiles)
    4. Return best Ap for each WA type
    """
    # Calculate WA features
    features = calculate_enhanced_wa(elog_persona, cutoff, Tp_sec)

    if features is None or features.height < 20:
        return None

    # Get actual churn labels
    future_start = cutoff + pl.duration(seconds=GRACE_SEC)
    future_end = future_start + pl.duration(seconds=CHURN_CHECK_WINDOW_SEC)

    future_activity = (
        elog_persona
        .filter(
            (pl.col("timestamp") > future_start) &
            (pl.col("timestamp") <= future_end)
        )
        .group_by("user_id")
        .agg(pl.len().alias("future_events"))
    )

    # Join features with labels
    eval_df = (
        features
        .join(future_activity, on="user_id", how="left")
        .with_columns([
            (pl.col("future_events").is_null() | (pl.col("future_events") == 0))
            .cast(pl.Int8).alias("actual_churn")
        ])
    )

    pdf = eval_df.to_pandas()
    y_true = pdf["actual_churn"].values

    # Test different WA types
    wa_types = ["WA_simple", "WA_trend", "WA_recency", "WA_enhanced", "WA_consistent"]

    results = []

    for wa_type in wa_types:
        wa_scores = pdf[wa_type].values

        best_f_beta = 0
        best_ap = None

        # Try different Ap values (percentiles)
        for percentile in AP_PERCENTILES:
            Ap = np.percentile(wa_scores, percentile)

            # Predict: churn if WA < Ap
            y_pred = (wa_scores < Ap).astype(int)

            # Calculate metrics
            TP = np.sum((y_true == 1) & (y_pred == 1))
            FP = np.sum((y_true == 0) & (y_pred == 1))
            FN = np.sum((y_true == 1) & (y_pred == 0))

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            # F-beta score
            f_beta = (1 + BETA**2) * precision * recall / ((BETA**2 * precision) + recall) \
                     if (precision + recall) > 0 else 0

            if f_beta > best_f_beta:
                best_f_beta = f_beta
                best_ap = Ap
                # ... store best metrics

        if best_ap is not None:
            results.append({...})

    return results
```

### ۴.۵.۳ حلقه‌ی اصلی Grid Search

```python
# File: enhanced_interpretable_churn_rule.py

all_results = []
personas = sorted(elog["persona"].unique())

for persona in personas:
    print(f"PERSONA {persona}")

    elog_persona = elog.filter(pl.col("persona") == persona)

    for Tp_days in TP_DAYS_RANGE:
        Tp_sec = Tp_days * 86400
        print(f"  Tp = {Tp_days} days:")

        window_results = []

        for cutoff in TRAIN_CUTOFFS:
            results = evaluate_ap_tp(elog_persona, persona, cutoff, Tp_sec)

            if results is not None:
                for r in results:
                    window_results.append({
                        **r,
                        'persona': persona,
                        'Tp_days': Tp_days,
                        'Tp_sec': Tp_sec,
                        'cutoff': cutoff.strftime('%Y-%m-%d'),
                    })

        if len(window_results) > 0:
            all_results.extend(window_results)

# Save all results
df_all = pd.DataFrame(all_results)
df_all.to_csv("enhanced_wa_all_results.csv", index=False)
```

### ۴.۵.۴ انتخاب بهترین پارامترها

```python
# File: enhanced_interpretable_churn_rule.py

best_params_list = []

for persona in personas:
    df_persona = df_all[df_all['persona'] == persona]

    # For each (Tp, wa_type), calculate median metrics across cutoffs
    summary = (
        df_persona
        .groupby(['Tp_days', 'Tp_sec', 'wa_type'])
        .agg({
            'f1': ['median', 'std', 'min', 'max'],
            'f_beta': ['median', 'std'],
            'precision': 'median',
            'recall': 'median',
            'auc': 'median',
            'Ap': 'median',
        })
        .reset_index()
    )

    # Select best based on median F-beta
    best_idx = summary['f_beta_median'].idxmax()
    best = summary.loc[best_idx]

    best_params_list.append({
        'persona': int(persona),
        'Tp_days': int(best['Tp_days']),
        'Tp_sec': float(best['Tp_sec']),
        'wa_type': best['wa_type'],
        'Ap': float(best['Ap_median']),
        'median_f1': float(best['f1_median']),
        'std_f1': float(best['f1_std']),
        # ...
    })

df_best = pd.DataFrame(best_params_list)
df_best.to_csv("enhanced_wa_best_params.csv", index=False)

                    Tp_sec=Tp_sec,
                    decay_lambda=decay_lambda,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma
                )
                
                # محاسبه و ارزیابی
                wa_df = compute_enhanced_wa_batch(eventlog, cutoff, config, persona)
                
                # انتخاب بهترین WA type
                for wa_type in ["WA_simple", "WA_trend", "WA_recency", "WA_enhanced"]:
                    # ... ارزیابی با Ap مختلف
                    pass
                
                results.append({...})
    
    return results
```

---

## ۴.۶ نتایج Enhanced WA

> **منبع داده:** `enhanced_wa_best_params.csv`

### ۴.۶.۱ بهترین پارامترها برای هر پرسونا

از فایل `enhanced_wa_best_params.csv` (نتایج واقعی):

| Persona | Tp (روز) | WA Type | Ap | Median F1 | Std F1 | Precision | Recall |
|---------|----------|---------|------|-----------|--------|-----------|--------|
| 0 | 270 | WA_trend | 0.461 | 0.532 | 0.089 | 0.423 | 0.759 |
| 1 | 270 | WA_trend | 0.669 | 0.599 | 0.108 | 0.475 | 0.835 |
| 2 | 240 | WA_trend | 0.283 | 0.617 | 0.095 | 0.501 | 0.791 |
| 3 | 270 | WA_recency | 0.125 | 0.564 | 0.095 | 0.443 | 0.774 |
| 4 | 270 | WA_recency | 0.143 | 0.602 | 0.082 | 0.473 | 0.881 |
| 5 | 270 | WA_recency | 0.135 | 0.599 | 0.080 | 0.471 | 0.848 |
| 6 | 270 | WA_simple | 0.263 | 0.609 | 0.105 | 0.475 | 0.818 |
| 7 | 270 | WA_trend | 0.533 | 0.648 | 0.125 | 0.507 | 0.891 |
| 9 | 270 | WA_recency | 0.333 | 0.667 | 0.067 | 0.525 | 0.913 |
| 12 | 270 | WA_enhanced | 0.193 | 0.654 | 0.135 | 0.525 | 0.898 |
| 16 | 270 | WA_recency | 0.246 | 0.673 | 0.075 | 0.534 | 0.921 |
| 18 | 90 | WA_recency | 0.420 | 0.795 | 0.075 | 0.715 | 0.889 |
| 19 | 270 | WA_recency | 0.374 | 0.733 | 0.054 | 0.601 | 0.943 |

### ۴.۶.۲ مشاهدات کلیدی

**۱. توزیع WA Types:**
```
wa_type         تعداد پرسوناها
─────────────────────────────
WA_recency      10
WA_trend        5
WA_simple       3
WA_enhanced     2
WA_consistent   0
```

**۲. میانگین عملکرد:**

| معیار | مقدار |
|-------|-------|
| Average Median F1 | 0.625 |
| Average Precision | 0.497 |
| Average Recall | 0.852 |
| Average AUC | 0.699 |

**۳. Tp غالب:**
- اکثر پرسوناها: **Tp = 270 روز** (۹ ماه)
- استثنا: پرسونای ۱۸ با **Tp = 90 روز** (سریع‌ترین Churn)

### ۴.۶.۳ مقایسه‌ی WA Types

از مقایسه‌ی انواع WA:

```
Performance by WA type:
═══════════════════════════════════════════════════════════
WA Type         Avg F1    Avg Precision    Avg Recall    N
───────────────────────────────────────────────────────────
WA_recency      0.636     0.504            0.867         10
WA_trend        0.599     0.478            0.826         5
WA_enhanced     0.654     0.525            0.898         2
WA_simple       0.609     0.475            0.818         3
═══════════════════════════════════════════════════════════
```

> **💡 نتیجه‌گیری:**
> 
> - **WA_recency** در اکثر پرسوناها بهترین است
> - **WA_enhanced** بهترین F1 را دارد ولی فقط در ۲ پرسونا انتخاب شد
> - **Recall بالا** (۸۵٪+) نشان می‌دهد سیستم اکثر Churnerها را پیدا می‌کند

---

## ۴.۷ Temporal Drift: تحلیل

> **منبع کد:** `temporal_drift_analysis.py`

### ۴.۷.۱ تعریف Eras

در کد واقعی پروژه، سه Era تعریف شده:

```python
# File: temporal_drift_analysis.py

# Era definitions (based on Stack Overflow usage patterns)
ERAS = {
    'pre_covid': (2016, 2019),      # Normal times
    'covid': (2020, 2021),           # Remote work surge
    'post_llm': (2022, 2024),        # ChatGPT/Copilot era
}

# Test years for temporal cross-validation
CUTOFF_YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
GRACE_DAYS = 90

def get_era(year):
    for era_name, (start, end) in ERAS.items():
        if start <= year <= end:
            return era_name
    return 'unknown'
```

### ۴.۷.۲ تابع ارزیابی زمانی

```python
# File: temporal_drift_analysis.py

def evaluate_persona_year(elog, persona, Ap, Tp_sec, cutoff_year, wa_type):
    """
    Evaluate a single persona for a single cutoff year
    """
    cutoff = pl.datetime(cutoff_year, 1, 1)
    window_start = cutoff - pl.duration(seconds=int(Tp_sec))

    # Only events in the window for this persona
    win = (
        elog
        .filter(pl.col("persona") == persona)
        .filter(
            (pl.col("timestamp") >= window_start) &
            (pl.col("timestamp") <= cutoff)
        )
    )

    active_users = win.select("user_id").unique()

    if active_users.height == 0:
        return None

    # Calculate WA
    wa = calculate_wa(win, Tp_sec, cutoff, wa_type)
    wa = wa.join(active_users, on="user_id", how="inner")

    # Predict churn: WA < Ap
    wa = wa.with_columns([
        (pl.col("WA") < Ap).cast(pl.Int8).alias("pred_churn")
    ])

    # Actual churn after grace period
    after = (
        elog
        .filter(pl.col("persona") == persona)
        .filter(
            (pl.col("timestamp") > cutoff + pl.duration(seconds=GRACE_SEC)) &
            (pl.col("timestamp") < cutoff + pl.duration(seconds=GRACE_SEC + 365 * 86400))
        )
        .group_by("user_id")
        .agg(pl.len().alias("cnt_after"))
    )

    eval_df = (
        wa.join(after, on="user_id", how="left")
          .with_columns([
              (pl.col("cnt_after").is_null()).cast(pl.Int8).alias("actual_churn")
          ])
    )

    # Calculate metrics
    # ... precision, recall, f1

    return {
        "persona": int(persona),
        "cutoff_year": cutoff_year,
        "Tp_days": int(Tp_sec / 86400),
        "Ap": float(Ap),
        "num_users": len(y_true),
        "actual_churn_rate": float(y_true.mean()),
        "pred_churn_rate": float(y_pred.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
```

### ۴.۷.۳ نتایج واقعی Baseline

از `temporal_baseline_results.csv` - نمونه‌ای از Drift در پرسونای ۰:

| سال | کاربران | Actual Churn Rate | Pred Churn Rate | F1 |
|-----|---------|-------------------|-----------------|-----|
| 2017 | 2,634 | 19.4% | 55.8% | 0.443 |
| 2018 | 2,471 | 24.7% | 58.4% | 0.506 |
| 2019 | 2,005 | 22.0% | 67.2% | 0.452 |
| 2020 | 1,865 | 24.9% | 67.5% | 0.478 |
| 2021 | 1,647 | 30.5% | 70.1% | 0.549 |
| 2022 | 1,431 | 30.3% | 73.3% | 0.535 |
| 2023 | 1,256 | 35.4% | 74.8% | **0.599** |

**مشاهدات:**
- **معیار Actual Churn Rate افزایش یافته:** از ۱۹.۴٪ به ۳۵.۴٪
- **تعداد کاربران کاهش یافته:** از ۲,۶۳۴ به ۱,۲۵۶
- **معیار F1 در نوسان:** بین ۰.۴۴ تا ۰.۶۰

---

## ۴.۸ مقابله با Temporal Drift

> **منبع کد:** `temporal_drift_analysis.py`

### ۴.۸.۱ رویکرد ۱: Baseline (بدون تنظیم)

```python
# File: temporal_drift_analysis.py

print("APPROACH 1: BASELINE (No Temporal Adjustment)")
print("Using original Ap/Tp values across all years...")

baseline_results = []

for persona, Ap in Ap_map.items():
    Tp_sec = Tp_map[persona]

    for year in CUTOFF_YEARS:
        result = evaluate_persona_year(elog, persona, Ap, Tp_sec, year, WA_METHOD)
        if result:
            baseline_results.append(result)

df_baseline = pd.DataFrame(baseline_results)
```

### ۴.۸.۲ رویکرد ۲: Year-Specific Threshold Adjustment

```python
# File: temporal_drift_analysis.py

print("APPROACH 2: YEAR-SPECIFIC THRESHOLD ADJUSTMENT")
print("Adjusting Ap threshold based on year...")

# Calculate year-specific multipliers based on baseline performance
year_multipliers = {}
baseline_year_f1 = df_baseline.groupby('cutoff_year')['f1'].median()
reference_f1 = baseline_year_f1.max()  # Best year as reference

for year, year_f1 in baseline_year_f1.items():
    if year_f1 > 0:
        # If F1 is lower, adjust threshold
        multiplier = np.sqrt(year_f1 / reference_f1)
        year_multipliers[year] = multiplier
    else:
        year_multipliers[year] = 1.0

adjusted_results = []

for persona, Ap in Ap_map.items():
    Tp_sec = Tp_map[persona]

    for year in CUTOFF_YEARS:
        adjusted_Ap = Ap * year_multipliers.get(year, 1.0)
        result = evaluate_persona_year(elog, persona, adjusted_Ap, Tp_sec, year, WA_METHOD)
        if result:
            result['year_multiplier'] = year_multipliers.get(year, 1.0)
            adjusted_results.append(result)
```

### ۴.۸.۳ رویکرد ۳: Era-Based Models

```python
# File: temporal_drift_analysis.py

print("APPROACH 3: ERA-BASED MODELS")
print("Training separate thresholds for different eras...")

# Group years into eras
df_baseline['era'] = df_baseline['cutoff_year'].apply(get_era)

# Calculate era-specific multipliers
era_multipliers = {}
era_f1 = df_baseline.groupby('era')['f1'].median()
reference_era_f1 = era_f1.max()

for era, era_f1_val in era_f1.items():
    if era_f1_val > 0:
        multiplier = np.sqrt(era_f1_val / reference_era_f1)
        era_multipliers[era] = multiplier
    else:
        era_multipliers[era] = 1.0

era_results = []

for persona, Ap in Ap_map.items():
    Tp_sec = Tp_map[persona]

    for year in CUTOFF_YEARS:
        era = get_era(year)
        adjusted_Ap = Ap * era_multipliers.get(era, 1.0)
        result = evaluate_persona_year(elog, persona, adjusted_Ap, Tp_sec, year, WA_METHOD)
        if result:
            result['era'] = era
            result['era_multiplier'] = era_multipliers.get(era, 1.0)
            era_results.append(result)
```

### ۴.۸.۴ نتایج مقایسه‌ی سه رویکرد

از `temporal_comparison_summary.csv` (نتایج واقعی):

| Approach | Mean F1 | Median F1 | F1 Std | F1 Range |
|----------|---------|-----------|--------|----------|
| Baseline (No Adjustment) | 0.536 | 0.545 | 0.107 | 0.528 |
| Year-Specific Adjustment | 0.537 | 0.548 | 0.107 | 0.541 |
| Era-Based Models | 0.537 | 0.547 | 0.107 | 0.536 |

**مشاهده مهم:** تفاوت بین سه رویکرد **ناچیز** است! این نشان می‌دهد:

1. مشکل Temporal Drift با ضرایب ساده قابل حل نیست
2. نیاز به رویکرد پیچیده‌تر (LSTM) داریم

### ۴.۸.۵ Year Multipliers واقعی

از `temporal_year_multipliers.csv`:

| سال | Multiplier | تفسیر |
|-----|------------|-------|
| 2017 | 0.838 | آستانه پایین‌تر |
| 2018 | 0.931 | نزدیک به نرمال |
| 2019 | 0.894 | کمی پایین‌تر |
| 2020 | 0.916 | شروع COVID |
| 2021 | 0.949 | اوج COVID |
| 2022 | 0.947 | شروع LLM |
| 2023 | **1.000** | مرجع (بهترین عملکرد) |

> **💡 نکته:**
> 
> ۲۰۲۳ بهترین سال است چون:
> - مدل روی داده‌های اخیر آموزش دیده
> - رفتار کاربران با الگوی آموزش مطابقت دارد
> 
> سال‌های قدیمی‌تر عملکرد بدتری دارند چون رفتار متفاوت بوده.

---

## ۴.۹ ارزیابی Temporal با WA_recency

> **منبع کد:** `simple_eval_temporal.py`

### ۴.۹.۱ اسکریپت ارزیابی زمانی

```python
# File: simple_eval_temporal.py

"""
Temporal-Aware Churn Evaluation Script

This script addresses temporal concept drift by adjusting Ap thresholds
based on the year to maintain consistent performance across time periods.
"""

# Configuration
WA_METHOD = 'WA_recency'  # Winner from comparison
CUTOFF_YEAR = 2016
GRACE_DAYS = 180

# Load temporal multipliers
year_multipliers_df = pd.read_csv("temporal_year_multipliers.csv")
year_multipliers = dict(zip(year_multipliers_df["year"], 
                            year_multipliers_df["multiplier"]))

# Get multiplier for current year
year_multiplier = year_multipliers.get(CUTOFF_YEAR, 1.0)

# Adjust Ap values based on year
Ap_map = {persona: Ap * year_multiplier for persona, Ap in Ap_map.items()}
```

### ۴.۹.۲ محاسبه‌ی WA در اسکریپت ارزیابی

```python
# File: simple_eval_temporal.py

# Calculate WA based on method
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
    ((pl.col("recent_gain") + 1.0) / (pl.col("old_gain") + 1.0)).alias("activity_trend"),
    ((-pl.col("days_since_last") / 30.0).exp()).alias("recency_weight"),
])

# Select WA calculation based on method
if WA_METHOD == 'WA_simple':
    wa_expr = pl.col("total_gain")
elif WA_METHOD == 'WA_trend':
    wa_expr = pl.col("total_gain") * pl.col("activity_trend").clip(0.5, 2.0)
elif WA_METHOD == 'WA_recency':
    wa_expr = pl.col("total_gain") * pl.col("recency_weight")
elif WA_METHOD == 'WA_enhanced':
    wa_expr = (
        pl.col("total_gain")
        * pl.col("activity_trend").clip(0.5, 2.0)
        * pl.col("recency_weight")
    )
```

### ۴.۹.۳ Era-Aware Calibration

> **منبع کد:** `simple_eval_v2.py`

```python
# File: simple_eval_v2.py

# Era helpers
def get_era(year: int) -> str:
    if 2016 <= year <= 2019:
        return "pre_covid"
    if 2020 <= year <= 2021:
        return "covid"
    return "post_llm"

# Era-aware calibration (per persona, per cutoff/era)
Ap_used = float(Ap_base)

if CALIBRATION_MODE == "era_quantile_clip":
    # Choose Ap_used at the actual_rate quantile of WA distribution
    if 0.0 < actual_rate < 1.0:
        q = float(np.quantile(wa_vals, actual_rate))
        if np.isfinite(q) and q > 0:
            Ap_used = q
            
            # Clip around Ap_base for stability
            lo = float(Ap_base) * CALIB_CLIP_MIN  # 0.7
            hi = float(Ap_base) * CALIB_CLIP_MAX  # 1.5
            Ap_used = float(np.clip(Ap_used, lo, hi))
```

**توضیح Calibration:**

| روش | فرمول | مزیت |
|-----|-------|------|
| `none` | $A_p = A_p^{base}$ | ساده |
| `era_quantile` | $A_p = Q_{actual\_rate}(WA)$ | تطابق با نرخ واقعی |
| `era_quantile_clip` | $A_p = \text{clip}(Q, 0.7 \times A_p^{base}, 1.5 \times A_p^{base})$ | پایدارتر |

---

## ۴.۱۰ خلاصه‌ی الگوریتم Enhanced WA

```
Algorithm: Enhanced WA Churn Prediction
══════════════════════════════════════════════════════════════

INPUT:
  - eventlog: Event Log
  - cutoff: Evaluation date
  - config: {persona: (Tp, Ap, wa_type, lambda, weights)}

OUTPUT:
  - predictions: {user_id: pred_churn}

PROCEDURE:
  1. FOR each persona p:
       a. Load config for persona p
       b. Define window: [cutoff - Tp, cutoff]
       c. FOR each user u in persona p:
            i.   Compute WA_simple(u)
            ii.  Compute WA_trend(u)
            iii. Compute WA_recency(u, lambda)
            iv.  Compute WA_enhanced(u, weights)
            v.   Select WA based on wa_type
            vi.  pred_churn(u) = 1 if WA(u) < Ap else 0
       d. Collect predictions

  2. RETURN predictions

IMPROVEMENT over Rule-based:
  - Trend detection: captures increasing/decreasing activity
  - Recency weighting: recent events matter more
  - Temporal adaptation: better drift resistance

══════════════════════════════════════════════════════════════
```

---

## ۴.۱۱ محدودیت‌های باقی‌مانده

با وجود بهبودها، هنوز محدودیت‌هایی داریم:

### ۴.۱۱.۱ پارامترهای دستی

| محدودیت | توضیح |
|---------|-------|
| $\lambda$ انتخابی | باید Grid Search شود |
| وزن‌ها ثابت | $\alpha, \beta, \gamma$ دستی |
| Ap per-persona | نیاز به تنظیم جداگانه |

### ۴.۱۱.۲ Feature Engineering محدود

سه نوع WA کافی نیست:
- **Pattern Detection:** burst‌های فعالیت
- **Seasonality:** الگوهای فصلی
- **Cross-user:** مقایسه با کاربران مشابه

### ۴.۱۱.۳ سقف عملکرد

$$
\max(F1_{\text{Enhanced WA}}) \approx 0.65-0.70
$$

برای عملکرد بهتر، نیاز به مدل‌های یادگیری عمیق داریم (فصل ۵).

> **💡 مثال مفهومی:**
> 
> Enhanced WA فقط **سه عدد** درباره‌ی هر کاربر محاسبه می‌کند:
> - چقدر فعالیت کرده؟
> - روندش چطور بوده؟
> - آخرین فعالیتش کی بوده؟
> 
> اما الگوهای پنهان را نمی‌بیند:
> - آیا کاربر **فقط آخر هفته** فعال است؟
> - آیا **بعد از رسیدن به ۱۰K Rep** فعالیتش کم شده؟
> - آیا **بعد از دریافت downvote** ناامید شده؟
> 
> برای این الگوها، LSTM لازم است (فصل ۵).

---

## ۴.۱۲ خلاصه‌ی فصل

در این فصل:

- ✅ مفهوم Temporal Drift توضیح داده شد
- ✅ سه نوع Enhanced WA: trend, recency, enhanced
- ✅ فرمول‌های ریاضی و کد واقعی از `enhanced_interpretable_churn_rule.py`
- ✅ Grid Search با ۸ مقدار Tp و ۱۷ مقدار Ap
- ✅ نتایج: **WA_recency** در ۱۰ از ۲۰ پرسونا بهترین است
- ✅ تحلیل Temporal Drift با `temporal_drift_analysis.py`
- ✅ سه رویکرد مقابله: Baseline، Year-Specific، Era-Based
- ✅ **نتیجه:** تفاوت ناچیز → نیاز به LSTM

### فایل‌های خروجی واقعی

| فایل | محتوا |
|------|-------|
| `enhanced_wa_all_results.csv` | همه‌ی ترکیبات تست‌شده |
| `enhanced_wa_best_params.csv` | بهترین پارامتر هر پرسونا |
| `temporal_baseline_results.csv` | Baseline بدون تنظیم |
| `temporal_year_adjusted_results.csv` | تنظیم سالانه |
| `temporal_era_based_results.csv` | تنظیم Era |
| `temporal_comparison_summary.csv` | مقایسه‌ی سه رویکرد |
| `temporal_year_multipliers.csv` | ضرایب سالانه |

### منابع کد

| فایل | محتوا |
|------|-------|
| `enhanced_interpretable_churn_rule.py` | محاسبه‌ی Enhanced WA و Grid Search |
| `temporal_drift_analysis.py` | تحلیل Drift و سه رویکرد |
| `simple_eval_temporal.py` | ارزیابی با تنظیم زمانی |
| `simple_eval_v2.py` | Era-aware Calibration |

**سؤال کلیدی فصل بعد:** آیا LSTM می‌تواند الگوهای پیچیده‌تر را یاد بگیرد و Temporal Drift را بهتر مدیریت کند؟
