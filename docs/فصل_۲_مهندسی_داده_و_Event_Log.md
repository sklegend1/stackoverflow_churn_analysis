<div dir="rtl" align="right">

# فصل ۲ — مهندسی داده و ساخت Event Log

---

## ۲.۱ مقدمه

در این فصل، فرآیند تبدیل داده‌های خام Stack Overflow به ساختاری قابل استفاده برای مدل‌های پیش‌بینی Churn شرح داده می‌شود. این فرآیند شامل:

1. انتقال داده از MySQL به Parquet
2. ساخت Event Log یکپارچه
3. استخراج Core Activities با XGBoost
4. تقسیم Train/Eval

---

## ۲.۲ زیرساخت داده

### ۲.۲.۱ انتخاب ابزار: DuckDB + Polars

در جلسه‌ی سوم پروژه، بحث مفصلی درباره‌ی انتخاب ابزار پردازش داده انجام شد:

> «با توجه به حجم بالای داده‌ها و محدودیت منابع سیستم، استفاده از Apache Spark نیاز به کلاستر دارد. DuckDB + Polars ترکیبی سبک‌وزن با قابلیت پردازش میلیون‌ها رکورد روی یک ماشین معمولی است.»

| ابزار | نقش | مزیت |
|-------|-----|------|
| **MySQL** | ذخیره‌سازی اولیه | دیتاست رسمی SO |
| **DuckDB** | کوئری‌های SQL روی فایل | Lazy، بدون Memory overhead |
| **Polars** | پردازش DataFrame | Rust-based، سریع |
| **Parquet** | فرمت ذخیره‌سازی | Columnar، فشرده، قابل Lazy scan |

### ۲.۲.۲ معماری کلی

```
MySQL (Docker)
    │
    ├── filtered_users → filtered_users_parquet/
    ├── filtered_posts → filtered_posts_parquet/
    ├── posthistory   → posthistory_parquet/
    └── comments      → comments_parquet/
              │
              ▼
        DuckDB/Polars (Lazy Scan)
              │
              ▼
        eventlog_v2.parquet
              │
              ▼
        core_eventlog_train_v1.parquet
        core_eventlog_eval_v1.parquet
```

---

## ۲.۳ ساخت Event Log

### ۲.۳.۱ تعریف Event Log

Event Log یک ساختار داده‌ای است که **تمام رویدادهای مرتبط با کاربران** را به‌صورت زمانی مرتب شده نگه می‌دارد:

| ستون | نوع | توضیح |
|------|-----|-------|
| `id` | Int64 | شناسه‌ی یکتا (row index) |
| `user_id` | Int64 | شناسه‌ی کاربر |
| `timestamp` | Datetime | زمان وقوع رویداد |
| `action_type_id` | Int32 | نوع اکشن (کدگذاری شده) |
| `post_id` | Int64 | شناسه‌ی پست مرتبط |
| `post_type_id` | Int32 | نوع پست (Question=1, Answer=2) |
| `persona` | Int32 | شماره‌ی پرسونا (0–19) |
| `gap_time_actions_seconds` | Float64 | فاصله از اکشن قبلی (ثانیه) |
| `gap_time_actions_days` | Float64 | فاصله از اکشن قبلی (روز) |

### ۲.۳.۲ منابع رویداد

Event Log از سه منبع ساخته می‌شود:

#### منبع ۱: ایجاد پست (Posts)

```python
CREATE_QUESTION_ACTION = 1
CREATE_ANSWER_ACTION = 2

events_create_post = (
    posts
    .join(users, left_on="owneruserid", right_on="user_id", how="inner")
    .with_columns([
        pl.when(pl.col("posttypeid") == 1)
          .then(pl.lit(CREATE_QUESTION_ACTION))
          .when(pl.col("posttypeid") == 2)
          .then(pl.lit(CREATE_ANSWER_ACTION))
          .otherwise(pl.lit(None))
          .alias("action_type_id"),
    ])
)
```

#### منبع ۲: تاریخچه‌ی پست (PostHistory)

جدول `posthistory` شامل تمام تغییرات روی پست‌هاست. هر `PostHistoryTypeId` به یک `action_type_id` داخلی نگاشت می‌شود:

```python
PH_ACTION_MAP = {
    4: 13,   # Edit Title
    5: 14,   # Edit Body
    6: 15,   # Edit Tags
    7: 16,   # Rollback Title
    8: 17,   # Rollback Body
    9: 18,   # Rollback Tags
    10: 30,  # Close Post
    11: 31,  # Reopen Post
    12: 32,  # Delete
    13: 33,  # Undelete
    14: 34,  # Lock
    15: 35,  # Unlock
    # ... و سایر موارد
}
```

**نکته مهم:** فقط رویدادهایی که `userid` آنها در لیست کاربران هدف (`filtered_users`) است، وارد Event Log می‌شوند.

#### منبع ۳: نظرات (Comments)

```python
COMMENT_ACTION = 70

events_comments = (
    comments
    .join(users, left_on="userid", right_on="user_id", how="inner")
    .with_columns([
        pl.lit(COMMENT_ACTION).alias("action_type_id"),
    ])
)
```

### ۲.۳.۳ ادغام و مرتب‌سازی

```python
eventlog_lazy = pl.concat(
    [events_create_post, events_ph, events_comments],
    how="diagonal_relaxed"
)

# مرتب‌سازی بر اساس کاربر و زمان
eventlog_lazy = eventlog_lazy.sort(["user_id", "timestamp"])
```

### ۲.۳.۴ محاسبه‌ی Gap Time

فاصله‌ی زمانی بین رویدادهای متوالی هر کاربر:

```python
eventlog_lazy = eventlog_lazy.with_columns([
    # Gap Time به ثانیه
    (pl.col("timestamp")
      .diff()
      .over("user_id") / 1_000_000)
      .alias("gap_time_actions_seconds"),
    
    # Gap Time به روز
    (pl.col("timestamp")
      .diff()
      .over("user_id") / (1_000_000 * 60 * 60 * 24))
      .alias("gap_time_actions_days"),
])
```

### ۲.۳.۵ خروجی نهایی

```python
eventlog = eventlog_lazy.collect(engine="streaming")
eventlog.write_parquet("eventlog_v2.parquet")
```

**حجم خروجی:** چندین میلیون رکورد

---

## ۲.۴ انواع Action Types

جدول کامل `ActionTypes.csv`:

| ActionTypeId | Name | Category |
|-------------|------|----------|
| 1 | CreateQuestion | Post Creation |
| 2 | CreateAnswer | Post Creation |
| 13 | EditTitle | Editing |
| 14 | EditBody | Editing |
| 15 | EditTags | Editing |
| 16 | RollbackTitle | Quality Control |
| 17 | RollbackBody | Quality Control |
| 18 | RollbackTags | Quality Control |
| 30 | ClosePost | Moderation |
| 31 | ReopenPost | Moderation |
| 32 | DeletePost | Moderation |
| 33 | UndeletePost | Moderation |
| 70 | Comment | Social |

---

## ۲.۵ استخراج Core Activities

### ۲.۵.۱ مسئله

تمام انواع Action برای تشخیص Churn اهمیت یکسان ندارند. برخی فعالیت‌ها:
- بسیار نادر هستند
- برای برخی پرسوناها اصلاً انجام نمی‌شوند
- نویز ایجاد می‌کنند

### ۲.۵.۲ رویکرد: XGBoost One-vs-Rest

برای هر پرسونا، یک مدل XGBoost آموزش می‌بینید که:

$$
\text{Target}_p = \begin{cases}
1 & \text{if persona} = p \\
0 & \text{otherwise}
\end{cases}
$$

سپس از **Feature Importance (Gain)** استفاده می‌کنیم تا مشخص شود کدام Action Types برای تشخیص آن پرسونا مهم‌تر هستند.

```python
for P in personas:
    # تعریف target به‌صورت One-vs-Rest
    df = fm.with_columns([
        (pl.col("persona") == P).cast(pl.Int8).alias("target")
    ])
    
    X = df.select(feature_cols).to_pandas()
    y = df.select("target").to_pandas()["target"]
    
    dtrain = xgb.DMatrix(X, label=y)
    
    params = {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.1,
    }
    
    model = xgb.train(params, dtrain, num_boost_round=200)
    importance = model.get_score(importance_type="gain")
```

### ۲.۵.۳ انتخاب Top-K Actions

برای هر پرسونا، **۷ اکشن با بیشترین Gain** انتخاب می‌شوند:

```python
core_top = (
    core
    .sort(["persona", "gain"], descending=[False, True])
    .group_by("persona")
    .head(7)
    .select(["persona", "action_type_id", "gain"])
)
```

**چرا ۷؟** این عدد بر اساس آزمایش تجربی انتخاب شده. تعداد کمتر باعث از دست رفتن اطلاعات می‌شود و تعداد بیشتر نویز اضافه می‌کند.

### ۲.۵.۴ مثال خروجی

| persona | action_type_id | action_name | gain |
|---------|---------------|-------------|------|
| 0 | 2 | CreateAnswer | 145.2 |
| 0 | 70 | Comment | 89.7 |
| 0 | 14 | EditBody | 34.5 |
| 1 | 1 | CreateQuestion | 178.3 |
| 1 | 70 | Comment | 56.2 |
| ... | ... | ... | ... |

---

## ۲.۶ ساخت Core Event Log

### ۲.۶.۱ فیلتر بر اساس Core Actions

```python
core_eventlog_lazy = (
    elog
    .join(core_top, on=["persona", "action_type_id"], how="inner")
)
```

این Join فقط رویدادهایی را نگه می‌دارد که:
- برای پرسونای مربوطه
- جزو ۷ اکشن مهم باشند

### ۲.۶.۲ تقسیم Train/Eval

تقسیم‌بندی به‌صورت **Deterministic** و **User-level** انجام می‌شود:

```python
SEED = 42
EVAL_RATIO = 0.20

# محاسبه‌ی عدد تصادفی deterministic
m = 2147483647  # 2^31-1
a = 1103515245

users = users.with_columns([
    (
        (
            (pl.col("user_id") + pl.col("persona") * 1000003 + SEED)
            * a
        ) % m
    ) / m
    .alias("rnd"),
])

users = users.with_columns([
    (pl.col("rnd") < EVAL_RATIO).alias("is_eval")
])
```

**چرا User-level؟** اگر رویدادهای یک کاربر بین Train و Eval تقسیم شوند، Data Leakage رخ می‌دهد.

### ۲.۶.۳ محاسبه‌ی Gap Time برای Core Actions

```python
def finalize_core_log(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.with_columns([
        # Gap Time بر اساس فقط Core Actions
        (
            pl.col("timestamp").diff().over("user_id") / 1_000_000
        ).alias("gap_time_key_actions_seconds"),
        
        (
            pl.col("timestamp").diff().over("user_id") / (1_000_000 * 60 * 60 * 24)
        ).alias("gap_time_key_actions_days"),
    ])
    return lf
```

### ۲.۶.۴ خروجی نهایی

```python
core_train.write_parquet("core_eventlog_train_v1.parquet")
core_eval.write_parquet("core_eventlog_eval_v1.parquet")
```

| فایل | تعداد رکورد | نقش |
|------|-------------|-----|
| `core_eventlog_train_v1.parquet` | ~۸۰٪ | آموزش |
| `core_eventlog_eval_v1.parquet` | ~۲۰٪ | ارزیابی |

---

## ۲.۷ Feature Matrix

### ۲.۷.۱ ساختار

Feature Matrix برای هر کاربر یک بردار ویژگی ایجاد می‌کند:

| Feature | توضیح |
|---------|-------|
| `count_action_{id}` | تعداد هر نوع اکشن |
| `gap_mean_action_{id}` | میانگین فاصله‌ی زمانی |
| `gap_std_action_{id}` | انحراف معیار فاصله |
| `persona` | شماره‌ی پرسونا (target) |

### ۲.۷.۲ استفاده

این ماتریس برای آموزش XGBoost در مرحله‌ی Feature Importance استفاده می‌شود.

---

## ۲.۸ ملاحظات عملی

### ۲.۸.۱ Memory Management

```python
eventlog = eventlog_lazy.collect(engine="streaming")
```

استفاده از `engine="streaming"` باعث می‌شود داده‌ها به‌صورت batch پردازش شوند و حافظه‌ی اصلی پر نشود.

### ۲.۸.۲ زمان اجرا

| مرحله | زمان تقریبی |
|-------|-------------|
| ساخت eventlog_v2 | ~۵ دقیقه |
| ساخت core_eventlog | ~۲ دقیقه |
| XGBoost Feature Importance | ~۱۰ دقیقه |

---

## ۲.۹ Schema نهایی

### eventlog_v2.parquet

```
id                       : Int64
user_id                  : Int64
timestamp                : Datetime[μs]
action_type_id           : Int32
post_id                  : Int64
post_type_id             : Int32
persona                  : Int32
LastAccessDate           : Datetime
AccountCreationDate      : Datetime
gap_time_actions_seconds : Float64
gap_time_actions_days    : Float64
```

### core_eventlog_train_v1.parquet

```
id                             : Int64
user_id                        : Int64
timestamp                      : Datetime[μs]
action_type_id                 : Int32
post_id                        : Int64
post_type_id                   : Int32
persona                        : Int32
gain                           : Float64
gap_time_key_actions_seconds   : Float64
gap_time_key_actions_days      : Float64
```

---

## ۲.۱۰ خلاصه‌ی فصل

در این فصل:

- ✅ معماری داده توضیح داده شد: MySQL → Parquet → Polars
- ✅ ساخت Event Log از سه منبع (Posts, PostHistory, Comments)
- ✅ محاسبه‌ی Gap Time برای تحلیل رفتار زمانی
- ✅ استخراج Core Activities با XGBoost One-vs-Rest
- ✅ تقسیم Deterministic و User-level به Train/Eval

**خروجی‌های اصلی:**
- `eventlog_v2.parquet` — رویدادهای کامل
- `core_eventlog_train_v1.parquet` — رویدادهای کلیدی (Train)
- `core_eventlog_eval_v1.parquet` — رویدادهای کلیدی (Eval)



</div>
