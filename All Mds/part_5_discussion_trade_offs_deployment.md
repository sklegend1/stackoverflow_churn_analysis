# Part 5 — Discussion, Trade-offs & Deployment

---

## 5.1 جمع‌بندی مسیر پروژه

در این پروژه، مسئله‌ی Churn نه به‌عنوان یک وضعیت ایستا، بلکه به‌عنوان یک **پدیده‌ی رفتاری–زمانی** مورد بررسی قرار گرفت. مسیر توسعه از یک سیستم تفسیری مبتنی بر قانون آغاز شد و به‌صورت تدریجی، داده‌محور و کنترل‌شده به یک مدل Sequence-based مبتنی بر LSTM تکامل یافت.

نکته‌ی کلیدی این مسیر آن است که:

> هر فاز، محدودیت‌های فاز قبل را آشکار کرده و مبنای منطقی فاز بعدی را فراهم کرده است.

به‌عبارت دیگر، LSTM جایگزین شکست‌خورده‌ی سیستم Rule-based نیست، بلکه نتیجه‌ی طبیعی بلوغ تحلیل مسئله است.

---

## 5.2 مقایسه نهایی: Rule-based vs LSTM

مقایسه‌ی دو رویکرد اصلی پروژه در جدول زیر ارائه شده است:

| Dimension | Rule-based System | LSTM-based Model |
|---------|------------------|-----------------|
| Interpretability | Very High | Low |
| Temporal Modeling | Approximate | Explicit |
| Handling Drift | Manual / Partial | Intrinsic |
| Feature Engineering | Heavy | Minimal |
| Accuracy (F1) | Medium–High | High |
| Stability over Time | Medium | High |
| Deployment Complexity | Low | High |
| Explainability for Stakeholders | Strong | Limited |

این جدول نشان می‌دهد که هیچ‌یک از دو سیستم به‌تنهایی بر دیگری غالب نیست؛ بلکه هر کدام برای هدف متفاوتی بهینه شده‌اند.

---

## 5.3 معماری پیشنهادی نهایی (Hybrid Strategy)

بر اساس نتایج تجربی و تحلیل trade-offها، معماری نهایی پیشنهادی به‌صورت **Hybrid** است:

- **Rule-based System** به‌عنوان لایه‌ی:
  - Explainability
  - Monitoring
  - Sanity Check
  - Decision Support

- **LSTM-based Model** به‌عنوان لایه‌ی:
  - Prediction Engine
  - Early Warning System
  - Production Scoring

در این معماری:

- خروجی LSTM سیگنال اصلی پیش‌بینی Churn است
- خروجی Rule-based برای توضیح «چرا» و اعتبارسنجی تصمیم استفاده می‌شود

---

## 5.4 استراتژی استقرار (Deployment Strategy)

### 5.4.1 Shadow Deployment

در فاز اول استقرار:

- مدل LSTM به‌صورت Shadow اجرا می‌شود
- تصمیم نهایی همچنان بر اساس Rule-based System اتخاذ می‌گردد
- اختلاف خروجی‌ها به‌صورت مداوم پایش می‌شود

این رویکرد ریسک عملیاتی را به حداقل می‌رساند.

---

### 5.4.2 Gradual Transition

پس از اطمینان از پایداری مدل:

- LSTM به‌عنوان منبع اصلی Score استفاده می‌شود
- Rule-based System نقش کنترلی و تفسیری را حفظ می‌کند

Thresholdهای تصمیم‌گیری می‌توانند به‌صورت پویا تنظیم شوند.

---

## 5.5 ریسک‌ها و ملاحظات

### 5.5.1 ریسک‌های Rule-based

- حساسیت به تغییرات رفتاری بلندمدت
- نیاز به بازتنظیم دوره‌ای Ap و Tp

### 5.5.2 ریسک‌های LSTM

- کاهش شفافیت تصمیم‌ها
- نیاز به مانیتورینگ مداوم Data Drift
- هزینه‌ی محاسباتی بالاتر

مدیریت این ریسک‌ها نیازمند ترکیب هر دو سیستم است.

---

## 5.6 مسیرهای توسعه‌ی آتی (Future Work)

مسیرهای بالقوه برای توسعه‌ی آتی پروژه:

- استفاده از Transformer-based Models برای توالی‌های طولانی‌تر
- افزودن Attention Mechanism برای افزایش Interpretability
- Online Learning برای انطباق سریع‌تر با Drift
- استفاده از Explainable AI tools برای تحلیل رفتار LSTM

---

## 5.7 نتیجه‌گیری نهایی

این پروژه نشان داد که:

- Churn یک مسئله‌ی ذاتاً Temporal است
- سیستم‌های تفسیری تا یک سقف مشخص قابل بهبود هستند
- مدل‌های Sequence-based برای داده‌های Non-stationary مناسب‌ترند

معماری نهایی پیشنهادی، تعادلی عملی بین **دقت، پایداری و تفسیرپذیری** ایجاد می‌کند و می‌تواند به‌عنوان یک چارچوب عمومی برای مسائل Churn مشابه مورد استفاده قرار گیرد.

---

**End of Part 5**
