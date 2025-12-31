# Part 1 — Data Engineering & Event Log

---

## 1.1 Objectives of the Data Engineering Phase

هدف اصلی این فاز، تبدیل داده‌های خام و ناهمگن Stack Overflow به یک **Event Log استاندارد، قابل پردازش و قابل بازتولید** بوده است؛ به‌گونه‌ای که:

- رفتار کاربران به‌صورت **زمان‌محور (Temporal)** قابل تحلیل باشد
- امکان محاسبه‌ی فاصله‌های زمانی بین رویدادها (Gap Times) فراهم شود
- داده‌ها برای هر دو مسیر **تفسیری (Rule-based)** و **مدل‌محور (Sequence-based)** قابل استفاده باشند
- محدودیت‌های سخت‌افزاری و منابع پروژه رعایت شود

بر همین اساس، این فاز به‌عنوان ستون فقرات کل پروژه طراحی و پیاده‌سازی شد.

---

## 1.2 Source Dataset Overview

### 1.2.1 Data Origin

داده‌های مورد استفاده از دیتاست رسمی **Stack Overflow** استخراج شده‌اند و شامل جداول زیر هستند:

- `users`
- `posts`
- `posthistory`
- `comments`
- `votes`

برای کنترل مقیاس، تنها کاربرانی که در پژوهش قبلی خوشه‌بندی شده‌اند (حدود **92 هزار کاربر**) در نظر گرفته شدند.

---

### 1.2.2 Persona Information

هر کاربر به یکی از **20 پرسونا** (خوشه‌ی رفتاری) اختصاص داده شده است. این پرسوناها:

- حاصل یک پژوهش قبلی مبتنی بر NMF هستند
- در تمام مراحل پروژه به‌عنوان **متغیر ساختاری اصلی** استفاده شده‌اند

تمام تحلیل‌ها، استخراج پارامترها و ارزیابی‌ها **persona-specific** انجام می‌شود.

---

## 1.3 Engineering Constraints and Design Decisions

### 1.3.1 Resource Constraints

پروژه با محدودیت‌های زیر مواجه بوده است:

- عدم دسترسی به Spark cluster یا زیرساخت سازمانی
- RAM محدود در محیط توسعه
- حجم بسیار بالای داده‌ها (ده‌ها میلیون ردیف)

بنابراین، معماری pipeline به‌صورت **File-based و Stage-wise** طراحی شد.

---

### 1.3.2 Tooling Selection

ابزارهای انتخاب‌شده:

- **Polars (Python)**: موتور اصلی ETL و پردازش DataFrame
- **DuckDB**: اجرای queryهای تحلیلی سریع روی فایل‌ها
- **Parquet**: فرمت ذخیره‌سازی اصلی

ابزار **Spark** عمداً در این فاز استفاده نشد، زیرا:

- هزینه و زمان راه‌اندازی بالایی دارد
- برای فاز exploration و preprocessing بیش‌ازحد سنگین است
- جایگزین‌های سبک‌تر (Polars/DuckDB) نیاز پروژه را پوشش می‌دهند

---

## 1.4 File-based Pipeline and Parquet Format

### 1.4.1 Why Parquet?

انتخاب Parquet به دلایل زیر انجام شد:

- فرمت ستونی (Columnar)
- سرعت بسیار بالا در scan و join
- مصرف حافظه‌ی کمتر نسبت به CSV
- پشتیبانی عالی توسط Polars و DuckDB

هر مرحله از pipeline خروجی مستقل خود را به‌صورت Parquet تولید می‌کند تا:

- pipeline قابل دیباگ باشد
- اجرای مجدد مراحل بدون تکرار کل فرآیند ممکن شود

---

## 1.5 Event Log Design

### 1.5.1 Why Event Log?

به‌جای snapshotهای کاربرمحور، از **Event Log** استفاده شد تا:

- توالی رفتار کاربران حفظ شود
- فاصله‌ی زمانی بین رویدادها قابل محاسبه باشد
- مدل‌سازی sequence-based ممکن شود

هر رویداد شامل حداقل این فیلدهاست:

- `user_id`
- `timestamp`
- `action_type_id`
- `persona`

---

### 1.5.2 Actor Consistency

یکی از نکات بحرانی در طراحی Event Log، تشخیص صحیح **actor** بود:

- در جدول `posthistory`، فیلد `UserId` انجام‌دهنده‌ی رویداد است
- در جدول `posts`، `OwnerUserId` مالک محتواست، نه لزوماً actor

بنابراین:

- رویدادهای PostHistory با `filtered_users` روی `UserId` join شدند
- جدول `posts` صرفاً برای enrichment (نوع پست) استفاده شد

این تصمیم مانع نسبت‌دادن اشتباه رفتارها به کاربران شد.

---

## 1.6 Event Sources and Action Mapping

### 1.6.1 Event Sources

Event Log نهایی از سه منبع ساخته شد:

1. **Posts**
   - Create Question
   - Create Answer

2. **PostHistory**
   - Edit
   - Rollback
   - Close
   - Reopen
   - سایر تغییرات محتوایی

3. **Comments**
   - Add Comment

جدول `votes` به‌دلیل نبود actor معتبر، عمداً از Event Log حذف شد.

---

### 1.6.2 Action Type Unification

تمام رویدادها به یک فضای یکپارچه‌ی `action_type_id` نگاشت شدند تا:

- مقایسه و وزن‌دهی اکشن‌ها ممکن شود
- feature engineering ساده‌تر گردد

---

## 1.7 Event Log Construction (Implementation)

### 1.7.1 Unified Event Log

پس از ساخت رویدادها از منابع مختلف:

- تمام eventها concatenate شدند
- بر اساس `(user_id, timestamp)` مرتب‌سازی شدند

نمونه اسکلت پیاده‌سازی:

```python
# Concatenate all event sources
eventlog_lazy = pl.concat(
    [events_posts, events_posthistory, events_comments],
    how="diagonal_relaxed"
)

# Sort by user and time
eventlog_lazy = eventlog_lazy.sort(["user_id", "timestamp"])
```

---

### 1.7.2 Gap Time Calculation

برای هر کاربر، فاصله‌ی زمانی بین رویدادهای متوالی محاسبه شد:

```python
eventlog_lazy = eventlog_lazy.with_columns([
    (pl.col("timestamp")
        .diff()
        .over("user_id")
        / 1_000_000
    ).alias("gap_time_actions_seconds")
])
```

این ویژگی پایه‌ی:

- استخراج ریتم فعالیت کاربران
- تعریف Tp در فازهای بعدی

را تشکیل می‌دهد.

---

## 1.8 Output of Phase 1

خروجی نهایی این فاز:

```
eventlog_v2.parquet
```

ویژگی‌های کلیدی خروجی:

- Event-based
- Persona-aware
- Sorted and gap-enriched
- Ready for:
  - Feature Matrix construction
  - Core Activity extraction
  - Sequence modeling (LSTM)

---

## 1.9 Role of This Phase in the Whole Project

این فاز:

- پیش‌نیاز تمام تحلیل‌های بعدی است
- هرگونه خطا در آن، downstream نتایج را مخدوش می‌کند
- یک‌بار به‌درستی انجام شده و در ادامه فقط **مصرف** می‌شود

به همین دلیل، بیشترین حساسیت مهندسی پروژه در این بخش متمرکز بوده است.

---

**End of Part 1**

