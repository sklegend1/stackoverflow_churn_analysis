<div dir="rtl">

# مدل خوشه‌بندی پرسونا برای کاربران جدید

## 📋 خلاصه

این سند فرآیند توسعه یک مدل یادگیری ماشین برای پیش‌بینی پرسونای کاربران جدید Stack Overflow را شرح می‌دهد. در تحقیقات اصلی، ۹۲,۲۸۹ کاربر با استفاده از روش Non-negative Matrix Factorization (NMF) به **۲۰ پرسونای متمایز** خوشه‌بندی شده بودند. از آنجا که به مدل خوشه‌بندی اصلی دسترسی نداریم، یک رویکرد یادگیری نظارت‌شده برای نگاشت کاربران جدید به این پرسوناهای موجود توسعه دادیم.

---

## 🎯 تعریف مسئله

### چالش‌ها
- مدل خوشه‌بندی NMF اصلی **در دسترس نیست**
- برخی ویژگی‌های استفاده شده در خوشه‌بندی اصلی برای کاربران جدید **قابل دسترسی نیستند** (مثل سن، برچسب نوع پرسونا)
- نیاز به پیش‌بینی پرسونا برای کاربرانی که در خوشه‌بندی اصلی نبودند
- باید با داده‌های موجود کار کند: پروفایل کاربران و لاگ فعالیت‌ها

### منابع داده موجود

| File | Description | Records |
|------|-------------|---------|
| `filtered_users_parquet/filtered_users_part_00000.parquet` | User profiles with cluster labels | 92,289 users |
| `eventlog_v2.parquet` | User activity events | 40.6M events |
| `nmf_cluster_statistics_20.csv` | Statistics for each cluster | 20 clusters × 105 features |

---

## 🔬 مسیر تحقیق

### گام اول: کاوش داده‌ها

آمار خوشه‌ها را تحلیل کردیم تا بفهمیم چه ویژگی‌هایی در خوشه‌بندی اصلی استفاده شده‌اند:

```
Key Features Identified:
├── User Profile Features
│   ├── Reputation (min, max, avg, median, std)
│   ├── UpVotes, DownVotes
│   ├── Views
│   └── AboutMe length
│
├── Activity Features
│   ├── Count of questions, answers, comments
│   ├── Post history (edits)
│   ├── Total activity duration
│   └── Collectivistic/Individualistic ratios
│
├── Demographic Features
│   ├── Geographic region (North America, Europe, South Asia, etc.)
│   ├── Age groups (18-24, 25-34, 35-44, etc.)
│   └── Persona types (H=Helper, C=Contributor, T=Teacher, I=Inquirer)
│
└── Behavioral Features
    └── Comment sentiment
```

### گام دوم: مهندسی ویژگی

از آنجا که دسترسی مستقیم به سن و برچسب نوع پرسونا برای کاربران جدید نداریم، روی **ویژگی‌های قابل استخراج** تمرکز کردیم:

#### ویژگی‌های پروفایل کاربر
- `reputation`, `upvotes`, `downvotes`, `views`
- `aboutme_length` (طول متن پاک‌شده از HTML)
- `total_duration` (عمر حساب به ماه)

#### ویژگی‌های فعالیت (از eventlog)
- `count_questions` (action_type_id = 1)
- `count_answers` (action_type_id = 2)
- `count_comments` (action_type_id = 70)
- `count_posthistory` (action_type_id = 13, 14, 15)
- `total_events` (همه فعالیت‌ها)

#### نسبت‌های مشتق‌شده
- `collectivistic_ratio` = answers / (answers + questions)
- `individualistic_ratio` = questions / (answers + questions)
- `answer_question_ratio` = answers / questions
- `reputation_per_answer`, `comments_per_qa` و غیره

#### منطقه جغرافیایی
استخراج از فیلد `Location` با استفاده از تطبیق کلمات کلیدی:
- North America, Europe, South Asia, East Asia
- Oceania, South America, Western Asia
- North Africa, Africa, Unknown

### گام سوم: توسعه مدل

چندین رویکرد را به صورت تکراری توسعه و آزمایش کردیم:

| Iteration | Model | Accuracy | Notes |
|-----------|-------|----------|-------|
| v1 | XGBoost Classifier | 57.9% | Basic features |
| v2 | Random Forest | 56.7% | Similar performance |
| v3 | Hybrid (XGBoost + Centroid) | 51.5% | Centroid matching didn't help |
| **v4** | **XGBoost (Class-balanced)** | **50.0%** | Production model with robust features |

### گام چهارم: بینش‌های کلیدی

#### تحلیل اهمیت ویژگی‌ها
ویژگی‌های برتر که پیش‌بینی خوشه را هدایت می‌کنند:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | region_South Asia | 20.7% |
| 2 | region_Europe | 17.9% |
| 3 | region_North America | 9.2% |
| 4 | region_Unknown | 5.2% |
| 5 | region_East Asia | 3.7% |
| 6 | count_answers | 1.7% |
| 7 | count_comments | 1.2% |
| 8 | action_70 (comments) | 1.8% |
| 9 | downvotes | 0.9% |
| 10 | reputation | 0.7% |

**یافته کلیدی**: منطقه جغرافیایی **عامل غالب** در تخصیص خوشه است و حدود ۶۰٪ اهمیت پیش‌بینی را تشکیل می‌دهد.

#### عملکرد به تفکیک خوشه
برخی خوشه‌ها با دقت بالا پیش‌بینی می‌شوند، برخی دیگر چالش‌برانگیز هستند:

| Cluster | Accuracy | Size | Dominant Characteristics |
|---------|----------|------|--------------------------|
| 1 | 80.6% | 13,515 | East Asia, 25-34 years, Helper |
| 9 | 65.7% | 5,254 | South Asia, 25-34 years |
| 4 | 62.1% | 9,739 | North America, 35-44 years |
| 5 | 48.8% | 18,131 | Europe, 25-34 years, Helper |
| 10 | 4.9% | 1,236 | South Asia, 25-34 years |

---

## 📦 فایل‌های ایجاد شده

| File | Size | Purpose |
|------|------|---------|
| `persona_predictor_production.joblib` | 47.2 MB | **مدل آموزش‌دیده** (از این استفاده کنید) |
| `persona_features_production.parquet` | 10.7 MB | ماتریس ویژگی برای ۹۲ هزار کاربر |
| `persona_feature_importance_production.csv` | 717 B | رتبه‌بندی اهمیت ویژگی‌ها |
| `persona_predictor_production.py` | 23.1 KB | اسکریپت آموزش و کلاس مدل |
| `predict_new_user_personas.py` | 8.3 KB | اسکریپت ابزار پیش‌بینی |

---

## 🚀 نحوه استفاده از مدل

### روش اول: Python API

```python
from predict_new_user_personas import predict_personas_for_users, get_cluster_description
import pandas as pd

# Load your new users data
new_users = pd.read_parquet('your_new_users.parquet')

# Optional: Load eventlog for activity features
eventlog = pd.read_parquet('eventlog_v2.parquet')

# Predict personas
results = predict_personas_for_users(new_users, eventlog)

# Results include:
# - user_id
# - predicted_cluster (0-19)
# - confidence (0-1)
# - top_3_clusters (list of most likely clusters)
# - prob_cluster_X (probability for each cluster)

print(results.head())
```

### روش دوم: خط فرمان

```bash
python predict_new_user_personas.py \
    --users new_users.parquet \
    --eventlog eventlog_v2.parquet \
    --output predictions.csv
```

### روش سوم: پیش‌بینی تک‌کاربره

```python
from predict_new_user_personas import predict_persona_single_user

user_data = {
    'Id': 12345,
    'Reputation': 5000,
    'UpVotes': 100,
    'DownVotes': 5,
    'Views': 1000,
    'Location': 'London, UK',
    'AboutMe': 'Software developer with 5 years of experience.',
    'CreationDate': '2020-01-01',
    'LastAccessDate': '2024-01-01'
}

result = predict_persona_single_user(user_data)
print(f"Predicted cluster: {result['predicted_cluster']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Top 3 clusters: {result['top_3_clusters']}")
```

### روش چهارم: دریافت توضیحات خوشه

```python
from predict_new_user_personas import get_cluster_description

# Get characteristics of cluster 1
desc = get_cluster_description(1)
print(f"Dominant region: {desc['dominant_region']}")
print(f"Dominant age: {desc['dominant_age']}")
print(f"Median reputation: {desc['reputation_median']}")
```

---

## 📊 درک پیش‌بینی‌ها

### امتیازات اطمینان

مدل برای هر پیش‌بینی یک امتیاز اطمینان (۰ تا ۱) خروجی می‌دهد:

| Confidence | Interpretation | Recommendation |
|------------|----------------|----------------|
| ≥ 0.7 | اطمینان بالا | به پیش‌بینی اعتماد کنید |
| 0.5 - 0.7 | اطمینان متوسط | پیش‌بینی احتمالاً درست است |
| 0.3 - 0.5 | اطمینان پایین | ۳ خوشه برتر را در نظر بگیرید |
| < 0.3 | اطمینان بسیار پایین | بررسی دستی نیاز است |

### استفاده از ۳ خوشه برتر

وقتی اطمینان پایین است، از فیلد `top_3_clusters` استفاده کنید:

```python
# For low-confidence predictions, consider all likely clusters
if result['confidence'] < 0.5:
    print(f"User could belong to clusters: {result['top_3_clusters']}")
```

---

## 🔧 راه‌های بهبود مدل

### ۱. افزودن ویژگی‌های بیشتر

#### الگوهای فعالیت زمانی
```python
# Extract time-based patterns from eventlog
- Hour of day distribution (working hours vs. off-hours)
- Day of week patterns (weekday vs. weekend activity)
- Activity trends over time (increasing, decreasing, stable)
- Response time to questions
```

#### تحلیل محتوا
```python
# Analyze post/comment content
- Primary programming language/technology tags
- Question vs. answer quality metrics
- Topic clustering (ML, web dev, databases, etc.)
- Writing style complexity
```

#### ویژگی‌های شبکه
```python
# User interaction patterns
- Number of distinct users interacted with
- Community participation breadth
- Badge diversity and types
```

### ۲. بهبود تشخیص جغرافیایی

استخراج منطقه فعلی از تطبیق کلمات کلیدی استفاده می‌کند. بهبودها:

```python
# Use a geocoding library
from geopy.geocoders import Nominatim

def get_region_advanced(location):
    geolocator = Nominatim(user_agent="persona_model")
    location = geolocator.geocode(location)
    if location:
        return map_country_to_region(location.address)
    return 'Unknown'
```

### ۳. مدیریت بهتر عدم توازن کلاس‌ها

۲۰ خوشه دارای اندازه‌های بسیار نامتوازن هستند (۵۵۲ تا ۱۸,۱۳۱ کاربر):

```python
# Options:
- SMOTE oversampling for minority clusters
- Focal loss for XGBoost
- Cluster hierarchical predictions (first predict region, then sub-cluster)
```

### ۴. رویکرد خوشه‌بندی سلسله‌مراتبی

با توجه به اهمیت جغرافیا، یک مدل دو مرحله‌ای را در نظر بگیرید:

```
Stage 1: Predict Geographic Region (high accuracy)
    ↓
Stage 2: Within-Region Cluster Prediction (more refined)
```

### ۵. یادگیری نیمه‌نظارت‌شده

از آمار خوشه‌ها برای ایجاد برچسب‌های شبه استفاده کنید:

```python
# For users close to cluster centroids, assign high-confidence labels
# Use these for semi-supervised training
from sklearn.semi_supervised import LabelPropagation
```

### ۶. روش‌های ترکیبی (Ensemble)

ترکیب چندین مدل:

```python
models = [
    XGBClassifier(...),
    LGBMClassifier(...),
    CatBoostClassifier(...),
    RandomForestClassifier(...)
]

# Voting or stacking ensemble
from sklearn.ensemble import VotingClassifier
ensemble = VotingClassifier(estimators=models, voting='soft')
```

### ۷. رویکرد یادگیری عمیق

برای دنباله‌های فعالیت از eventlog:

```python
# LSTM/Transformer for activity sequence modeling
- Input: Sequence of (action_type, timestamp, post_type)
- Output: Cluster prediction

# This could capture behavioral patterns better than aggregate features
```

### ۸. یادگیری فعال (Active Learning)

برای بهبود مستمر:

```python
# Identify uncertain predictions for human labeling
uncertain_users = results[results['confidence'] < 0.4]

# After manual labeling, retrain model
model.fit(updated_features, updated_labels)
```

---

## 📈 خلاصه معیارهای عملکرد

| Metric | Value | Notes |
|--------|-------|-------|
| Overall Accuracy | 50.0% | 20 classes, random = 5% |
| F1 (weighted) | 51.2% | Accounts for class imbalance |
| F1 (macro) | 37.2% | Equal weight per class |
| CV Accuracy | 50.0% ± 0.25% | Stable across folds |
| Best Cluster Accuracy | 80.6% | Cluster 1 |
| Worst Cluster Accuracy | 4.5% | Cluster 14 |

---

## 🧪 آزمایش‌های اهمیت ویژگی

### سؤال: آیا واقعاً منطقه جغرافیایی غالب است؟

در ابتدا مشاهده کردیم که ویژگی‌های منطقه حدود ۶۷٪ اهمیت ویژگی مدل XGBoost را تشکیل می‌دهند. این نگرانی را ایجاد کرد: آیا مدل فقط جغرافیا را یاد می‌گیرد، یا ویژگی‌های رفتاری معنادار هستند؟

ما یک مقایسه سیستماتیک با استفاده از سه مدل مختلف انجام دادیم:

### نتایج مقایسه Model ها

| Model | Clusters | Accuracy | Features Used |
|-------|----------|----------|---------------|
| **Region-ONLY** | 20 | **14.2%** | 10 (only region one-hot) |
| **Region + Behavior** | 20 | **48.9%** | ~60 (all features) |
| **Behavior-ONLY** | 8 super-clusters | 30.8% | ~50 (no region) |

### Key Insight

```
┌─────────────────────────────────────────────────────────────────────┐
│ Region-ONLY Accuracy:     14.2%                                     │
│ Region+Behavior Accuracy: 48.9%                                     │
│ Behavior Contribution:    +34.7% (HELPFUL!)                         │
└─────────────────────────────────────────────────────────────────────┘
```

### این به چه معناست

1. **منطقه به تنهایی کافی نیست** (۱۴.۲٪ به سختی بالاتر از تصادفی برای ۲۰ کلاس = ۵٪ است)
2. **ویژگی‌های رفتاری حیاتی هستند** - آنها ۳۴.۷٪ دقت اضافه می‌کنند!
3. **اهمیت ۶۷٪ منطقه گمراه‌کننده بود** - XGBoost از منطقه به عنوان *عبارت تعاملی* با رفتار استفاده می‌کند
4. خوشه‌بندی NMF اصلی **تعاملات منطقه × رفتار** را ثبت کرده است، نه فقط جغرافیا

### چرا منطقه "اهمیت" بالا اما دقت تنهای پایین نشان می‌دهد

اهمیت ویژگی XGBoost اندازه‌گیری می‌کند که یک ویژگی چند بار برای تقسیم استفاده می‌شود، نه قدرت پیش‌بینی مستقل آن. مدل الگوهایی مانند این را یاد می‌گیرد:

- "کاربران از جنوب آسیا که مکرراً پاسخ می‌دهند" → خوشه ۹
- "کاربران از اروپا که مکرراً پاسخ می‌دهند" → خوشه ۵
- "کاربران از جنوب آسیا که سؤال می‌پرسند" → خوشه ۱۰

بدون ویژگی‌های رفتاری، مدل نمی‌تواند بین انواع مختلف کاربر در همان منطقه تمایز قائل شود.

### Available Models

| Model File | Description | When to Use |
|------------|-------------|-------------|
| `persona_model_region_only.joblib` | Only geography | Testing/comparison |
| `persona_model_region_based.joblib` | Full features | **Production (recommended)** |
| `persona_model_behavior_based.joblib` | No region, 8 super-clusters | Cross-regional comparison |

### Super-Cluster Mapping (Behavior-Only Model)

هنگام استفاده از Behavior-Only Model، ۲۰ Cluster به ۸ Super-Cluster رفتاری نگاشت می‌شوند:

| Super-Cluster | Name | Original Clusters |
|---------------|------|-------------------|
| 0 | Moderate Contributors | [14, 8] |
| 1 | Active Mid-Level Users | [3, 18] |
| 2 | Casual Helpers | [15, 7, 5] |
| 3 | Regular Helpers | [10, 1, 6] |
| 4 | Elite Contributors | [12, 2] |
| 5 | Active Specialists | [16, 11] |
| 6 | Mid-Level Regulars | [0, 4, 9, 13, 19] |
| 7 | Newcomers | [17] |

---

## 🏁 نتیجه‌گیری

ما با موفقیت یک مدل پیش‌بینی پرسونا ساختیم که:

✅ کاربران جدید را به ۲۰ پرسونای موجود نگاشت می‌کند  
✅ دقت ۵۰٪ دارد (۱۰ برابر بهتر از تصادفی)  
✅ امتیازات اطمینان برای کیفیت پیش‌بینی ارائه می‌دهد  
✅ استفاده آسان از طریق Python API یا خط فرمان  

### Key Findings از آزمایش‌ها

✅ **Behavior نویز نیست** - ۳۴.۷٪+ Accuracy اضافه می‌کند  
✅ **Region × Behavior Interaction** - مدل Pattern های ترکیبی یاد می‌گیرد  
✅ **بهترین Model از هر دو استفاده می‌کند** - Region+Behavior با ۴۸.۹٪ Accuracy  

### Key Limitations

⚠️ نمی‌توان Age/Persona-Type Feature ها را از Clustering اصلی بازیابی کرد  
⚠️ برخی Cluster ها (۱۰، ۱۴) به دلیل Overlap ضعیف Predict می‌شوند  
⚠️ Behavior-Only Model فقط ۳۰.۸٪ Accuracy روی ۸ Super-Cluster دارد  

### استفاده پیشنهادی

برای ادغام با پیش‌بینی ریزش:
1. پرسوناها را برای کاربران جدید با این مدل پیش‌بینی کنید
2. بر اساس اطمینان ≥ ۰.۵ فیلتر کنید برای تخصیص‌های قابل اعتماد
3. برای کاربران با اطمینان پایین، ۳ خوشه برتر را در نظر بگیرید
4. به صورت دوره‌ای پیش‌بینی‌ها را با رفتار کاربر اعتبارسنجی کنید

---

## 📚 مراجع

- داده‌های آموزش: ۹۲,۲۸۹ کاربر Stack Overflow (۲۰۰۸-۲۰۲۴)
- خوشه‌بندی اصلی: NMF با ۲۰ مؤلفه
- داده‌های فعالیت: ۴۰.۶ میلیون رویداد در ۲۶ نوع عملیات
- مدل: XGBoost Classifier با وزن‌های کلاس متوازن

---

*تاریخ تولید سند: ۲ دی ۱۴۰۴*

</div>
