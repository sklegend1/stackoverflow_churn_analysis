# Part 2 — Interpretability-first Churn System

---

## 2.1 Motivation and Design Principles

هدف اصلی در این فاز، طراحی یک سیستم پیش‌بینی ریزش با ویژگی‌های زیر بوده است:

- **تفسیرپذیری کامل (Full Interpretability)**: امکان توضیح اینکه چرا یک کاربر در معرض Churn قرار گرفته است
- **پرسونا-محور بودن**: رفتار کاربران در خوشه‌های مختلف ذاتاً متفاوت است و نباید با یک قانون واحد مدل شود
- **وابستگی مستقیم به رفتار واقعی**: بدون استفاده از لیبل‌های قراردادی یا وضعیت‌های صوری
- **قابلیت ارزیابی مقاوم در زمان**: نتایج نباید به یک نقطه‌ی زمانی خاص وابسته باشند

بر این اساس، رویکرد این فاز به‌صورت **Formula-first** و **Data-driven parameter estimation** تعریف شد.

---

## 2.2 Formal Churn Rule Definition

برای هر کاربر \( u \) در پرسونا \( p \)، سیگنال Churn به‌صورت زیر تعریف می‌شود:

```
if WeightedActivity(u, Tp) < Ap:
    churn = 1
else:
    churn = 0
```

که در آن:

- **Tp**: طول پنجره‌ی مشاهده‌ی فعالیت (persona-specific)
- **Ap**: آستانه‌ی حداقلی فعالیت مؤثر (persona-specific)
- **WeightedActivity (WA)**: مجموع وزن‌دار فعالیت‌های کلیدی کاربر در بازه‌ی زمانی

این قانون عمداً ساده نگه داشته شده است تا:
- رفتار سیستم قابل توضیح باشد
- امکان استخراج داده‌محور پارامترها فراهم شود
- وابستگی به مدل‌های پیچیده کاهش یابد

---

## 2.3 Core Event Log and Activity Weighting

### 2.3.1 Core Activities

پس از ساخت Event Log کامل، مشخص شد که همه‌ی اکشن‌ها برای تشخیص رفتار کاربر ارزش یکسانی ندارند.

بنابراین برای هر پرسونا:

- یک مدل **XGBoost One-vs-Rest** آموزش داده شد
- Feature importance از نوع **Gain** استخراج گردید
- تنها Top-K اکشن‌های هر پرسونا به‌عنوان **Core Activities** انتخاب شدند

خروجی این مرحله:

- فایل `core_activity_per_persona_named.csv`
- نگاشت (persona, action_type_id → gain)

---

### 2.3.2 Core Event Log

Core Event Log زیرمجموعه‌ای از Event Log کامل است که فقط شامل:

- اکشن‌های کلیدی هر پرسونا
- کاربران همان پرسونا

می‌باشد. برای این لاگ:

- رویدادها بر اساس `(user_id, timestamp)` مرتب شدند
- فاصله‌ی زمانی بین رویدادهای کلیدی محاسبه شد

ویژگی‌های مهم:

- `gain`
- `gap_time_key_actions_seconds`
- `persona`

خروجی:

```
core_eventlog_v1.parquet
```

---

## 2.4 Definition of Weighted Activity (WA)

### 2.4.1 Baseline Definition

تعریف پایه‌ی Weighted Activity به‌صورت زیر است:

\[
WA(u, Tp) = \sum_{e \in window(Tp)} gain(e)
\]

که در آن:
- پنجره‌ی زمانی: \([cutoff - Tp, cutoff]\)
- فقط رویدادهای Core در نظر گرفته می‌شوند

این تعریف:
- غیرمنفی است
- ساده و قابل محاسبه است
- پایه‌ی تمام نسخه‌های بعدی WA محسوب می‌شود

---

### 2.4.2 Variants of WA

برای بهبود حساسیت سیستم به تغییرات رفتاری، نسخه‌های مختلفی از WA بررسی شد:

| Variant | Description |
|-------|-------------|
| WA_simple | مجموع gain بدون تعدیل |
| WA_trend | تعدیل بر اساس نسبت فعالیت اخیر به گذشته |
| WA_recency | وزن‌دهی نمایی بر اساس زمان آخرین فعالیت |
| **WA_enhanced** | ترکیب همزمان Magnitude × Trend × Recency |

نسخه‌ی نهایی انتخاب‌شده:

\[
WA_{enhanced} = WA_{simple} \times Trend \times Recency
\]

که در آن:
- **Trend** افزایش یا کاهش شدت فعالیت را نشان می‌دهد
- **Recency** فاصله‌ی زمانی از آخرین فعالیت را مدل می‌کند

این تعریف هنوز **قابل توضیح انسانی** است و پیچیدگی غیرضروری اضافه نمی‌کند.

---

## 2.5 Estimation of Tp (Observation Window)

### 2.5.1 Persona-specific Scaling

به‌جای انتخاب یک Tp ثابت، برای هر پرسونا مقدار Tp به‌صورت زیر پارامتریزه شد:

\[
Tp = \alpha \times p95(\text{gap\_time\_key\_actions})
\]

که در آن:
- `p95` صدک ۹۵ فاصله‌های زمانی بین اکشن‌های کلیدی است
- \( \alpha \) یک ضریب قابل جست‌وجو است

این روش باعث می‌شود Tp:
- با ریتم طبیعی هر پرسونا تطبیق یابد
- از انتخاب‌های دلخواه جلوگیری شود

---

## 2.6 Estimation of Ap (Activity Threshold)

### 2.6.1 Joint Search with XGBoost

برای هر ترکیب `(persona, alpha, cutoff)`:

- WA_enhanced به‌عنوان feature محاسبه شد
- لیبل واقعی Churn از آینده‌ی `cutoff + grace_period` ساخته شد
- یک مدل **XGBoost Classifier** آموزش داده شد

آستانه‌ی Ap به‌صورت:
- threshold روی احتمال خروجی مدل
- با استفاده از **precision-recall curve**

انتخاب گردید.

---

## 2.7 Sliding Cutoff Evaluation

برای جلوگیری از وابستگی نتایج به یک نقطه‌ی زمانی خاص:

- چندین cutoff روی timeline انتخاب شد (سالانه / شش‌ماهه)
- برای هر cutoff فرآیند کامل تکرار شد
- معیار نهایی به‌صورت **Median(F1)** گزارش شد

مزایا:
- مقاوم در برابر outlierهای زمانی
- مناسب برای داده‌های غیرایستا

---

## 2.8 Eligibility Filtering

در ارزیابی نهایی، فقط کاربرانی وارد محاسبه شدند که:

- حداقل یک Core Event در پنجره‌ی `[cutoff - Tp, cutoff]` داشته باشند

این فیلتر:
- از تورش مصنوعی metrics جلوگیری می‌کند
- ارزیابی را معنادار می‌سازد

---

## 2.9 Phase 2 Outcome

خروجی نهایی این فاز:

- یک سیستم **Rule-based، persona-specific و قابل توضیح**
- استفاده از **WA_enhanced** به‌عنوان بهترین نمایش فعالیت
- پارامترهای `(Tp, Ap)` مقاوم در زمان

محدودیت باقی‌مانده:

- وابستگی ذاتی به آستانه‌های ثابت
- حساسیت ساختاری به تغییرات رفتاری بلندمدت (Temporal Drift)

این محدودیت، انگیزه‌ی ورود به فاز مدل‌محور (LSTM) را فراهم کرد.

---

**End of Part 2**

