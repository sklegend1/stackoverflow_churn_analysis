<div dir="rtl" align="right">

# فصل ۵ — مدل LSTM برای پیش‌بینی Churn

---

## ۵.۱ مقدمه

در فصول قبل، محدودیت‌های سیستم Rule-based مشخص شد:
- پارامترهای ثابت
- عدم توجه به توالی رویدادها
- حساسیت به Temporal Drift

در این فصل، مدل **LSTM چندوظیفه‌ای (Multi-Task LSTM)** معرفی می‌شود که:
- توالی رویدادهای کاربر را پردازش می‌کند
- به Era (دوره‌ی زمانی) توجه دارد
- همزمان Churn، $A_p$ و $T_p$ را پیش‌بینی می‌کند

---

## ۵.۲ فرمول‌بندی مسئله

### ۵.۲.۱ ورودی‌ها

برای کاربر $u$ با پرسونا $p$ در زمان $t_c$:

| ورودی | شکل | توضیح |
|-------|-----|-------|
| Event Sequence | $(50, 2)$ | ۵۰ رویداد آخر: [gain, time_delta] |
| Global Features | $(4,)$ | ویژگی‌های زمانی |
| Persona Features | $(10,)$ | آمار پرسونا |

### ۵.۲.۲ خروجی‌ها

| خروجی | دامنه | توضیح |
|-------|-------|-------|
| $\hat{y}_{churn}$ | $(0, 1)$ | احتمال Churn |
| $\hat{A}_p$ | $(0, 1)$ | آستانه‌ی فعالیت |
| $\hat{T}_p$ | $[0, \infty)$ | پنجره‌ی زمانی (روز) |

---

## ۵.۳ معماری مدل

### ۵.۳.۱ نمای کلی

**معماری مدل Multi-Task LSTM:**

| لایه | ورودی | خروجی | توضیح |
|-----|-------|--------|------|
| Event Sequence | (batch, 50, 2) | → | gain + time_delta |
| LSTM Layers | (batch, 50, 2) | (batch, 128) | 2 لایه |
| Global Features | (batch, 4) | (batch, 4) | Context زمانی |
| Persona Features | (batch, 10) | (batch, 32) | Embedding |
| Concatenate | 128+4+32 | (batch, 164) | ترکیب |
| FC Shared | (batch, 164) | (batch, 128) | 256→128 |
| Churn Head | (batch, 128) | P(Churn) | 128→64→1 + Sigmoid |
| Ap Head | (batch, 128) | Ap | 128→64→1 + Sigmoid |
| Tp Head | (batch, 128) | Tp | 128→64→1 + ReLU |

### ۵.۳.۲ تعداد پارامترها

| بخش | تعداد پارامتر |
|-----|--------------|
| Persona Embedding | 352 |
| LSTM Layer 1 | 67,072 |
| LSTM Layer 2 | 131,584 |
| FC Shared | ~75,000 |
| Heads | ~25,000 |
| **مجموع** | **~165,000** |

---

## ۵.۴ لایه‌ها به تفصیل

### ۵.۴.۱ Event Sequence (ورودی)

آخرین ۵۰ رویداد کاربر قبل از $t_c$:

$$
\mathbf{X}^{(u)} = \begin{bmatrix} g_1 & \Delta t_1 \\ g_2 & \Delta t_2 \\ \vdots & \vdots \\ g_{50} & \Delta t_{50} \end{bmatrix} \in \mathbb{R}^{50 \times 2}
$$

که در آن:
- $g_i$: مقدار gain رویداد $i$
- $\Delta t_i = (t_i - t_{i-1}) / 86400$: فاصله‌ی زمانی به روز

**Padding:** برای کاربران با کمتر از ۵۰ رویداد:

$$
\mathbf{X}_{padded} = [\underbrace{0, ..., 0}_{50-n}, x_1, x_2, ..., x_n]
$$

### ۵.۴.۲ Global Features

ویژگی‌های زمانی سراسری:

$$
\mathbf{g}^{(u)} = \begin{bmatrix} d_{recency} \\ e_{era} \\ m_{norm} \\ y_{norm} \end{bmatrix}
$$

| ویژگی | فرمول | دامنه |
|-------|-------|-------|
| Days since last | $(t_c - t_{last}) / 86400$ | $[0, \infty)$ |
| Era | $\{0, 1, 2\}$ | Pre-COVID, COVID, Post-LLM |
| Month | $\text{month}(t_c) / 12$ | $[0.08, 1]$ |
| Year | $(\text{year}(t_c) - 2016) / 7$ | $[0, 1]$ |

**تعریف Era:**

| Era | سال‌ها | توضیح |
|-----|--------|-------|
| Pre-COVID (0) | 2016–2019 | رفتار پایه |
| COVID (1) | 2020–2021 | افزایش فعالیت آنلاین |
| Post-LLM (2) | 2022–2024 | تأثیر ChatGPT |

### ۵.۴.۳ Persona Features

ویژگی‌های آماری هر پرسونا:

$$
\mathbf{p}^{(k)} = \begin{bmatrix} \log(1 + \bar{r}_k) \\ \log(1 + \bar{u}_k) \\ \log(1 + \bar{a}_k) \\ \log(1 + \bar{q}_k) \\ \bar{d}_k / 190 \\ \rho_{coll,k} \\ \rho_{indiv,k} \\ \mathbb{1}[H_k] \\ \mathbb{1}[C_k] \\ \mathbb{1}[T_k] \end{bmatrix}
$$

که در آن:
- $\bar{r}_k$: میانگین reputation
- $\bar{u}_k$: میانگین upvotes
- $\bar{a}_k$: میانگین تعداد پاسخ
- $\bar{q}_k$: میانگین تعداد سؤال
- $\bar{d}_k$: میانگین مدت عضویت
- $\rho_{coll}$: نسبت رفتار جمع‌گرایانه
- $\rho_{indiv}$: نسبت رفتار فردگرایانه
- $H, C, T$: فلگ‌های Helper, Curator, Teacher

### ۵.۴.۴ Persona Embedding

تبدیل ویژگی‌های خام به فضای نهان:

$$
\mathbf{e}^{(u)} = \text{ReLU}(\mathbf{W}_e \cdot \mathbf{p}^{(k)} + \mathbf{b}_e)
$$

```python
self.persona_embed = nn.Linear(10, 32)
```

---

## ۵.۵ لایه‌ی LSTM

### ۵.۵.۱ ریاضیات LSTM

برای هر timestep $t$:

**Forget Gate:**
$$\mathbf{f}_t = \sigma(\mathbf{W}_f \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f)$$

**Input Gate:**
$$\mathbf{i}_t = \sigma(\mathbf{W}_i \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i)$$

**Candidate Cell:**
$$\tilde{\mathbf{c}}_t = \tanh(\mathbf{W}_c \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c)$$

**Cell Update:**
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Output Gate:**
$$\mathbf{o}_t = \sigma(\mathbf{W}_o \cdot [\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o)$$

**Hidden State:**
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

### ۵.۵.۲ پیکربندی

```python
self.lstm = nn.LSTM(
    input_size=2,         # gain + time_delta
    hidden_size=128,      # ابعاد hidden
    num_layers=2,         # دو لایه
    batch_first=True,
    dropout=0.2           # بین لایه‌ها
)
```

### ۵.۵.۳ خروجی

از **آخرین hidden state** لایه‌ی آخر استفاده می‌شود:

$$\mathbf{h}_{seq} = \mathbf{h}^{(L=2)}_{T=50} \in \mathbb{R}^{128}$$

---

## ۵.۶ ترکیب ویژگی‌ها

```python
combined = torch.cat([sequence_encoding, global_feat, persona_embed], dim=1)
```

$$\mathbf{z} = [\mathbf{h}_{seq} \| \mathbf{g} \| \mathbf{e}] \in \mathbb{R}^{164}$$

| بخش | ابعاد | اطلاعات |
|-----|-------|---------|
| $\mathbf{h}_{seq}$ | 128 | الگوهای زمانی |
| $\mathbf{g}$ | 4 | Context زمانی |
| $\mathbf{e}$ | 32 | Embedding پرسونا |

---

## ۵.۷ لایه‌های Shared

```python
self.fc_shared = nn.Sequential(
    nn.Linear(164, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
)
```

$$\mathbf{a}_1 = \text{Dropout}_{0.3}(\text{ReLU}(\mathbf{W}_1 \cdot \mathbf{z} + \mathbf{b}_1))$$
$$\mathbf{a}_2 = \text{Dropout}_{0.2}(\text{ReLU}(\mathbf{W}_2 \cdot \mathbf{a}_1 + \mathbf{b}_2))$$

---

## ۵.۸ سرهای خروجی (Prediction Heads)

### ۵.۸.۱ Churn Head

```python
self.churn_head = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
```

خروجی: $\hat{y}_{churn} \in (0, 1)$

### ۵.۸.۲ Ap Head

```python
self.ap_head = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
```

خروجی: $\hat{A}_p \in (0, 1)$

### ۵.۸.۳ Tp Head

```python
self.tp_head = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.ReLU()  # مثبت باشد
)
```

خروجی: $\hat{T}_p \in [0, \infty)$ (روز)

---

## ۵.۹ تابع هزینه

### ۵.۹.۱ Multi-Task Loss

$$\mathcal{L}_{total} = \lambda_1 \cdot \mathcal{L}_{churn} + \lambda_2 \cdot \mathcal{L}_{Ap} + \lambda_3 \cdot \mathcal{L}_{Tp}$$

| Task | Loss | $\lambda$ | توضیح |
|------|------|-----------|-------|
| Churn | BCE | 2.0 | هدف اصلی |
| Ap | MSE | 1.0 | آستانه |
| Tp | MSE (scaled) | 0.1 | پنجره (مقیاس بزرگ) |

### ۵.۹.۲ Binary Cross-Entropy

$$\mathcal{L}_{churn} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### ۵.۹.۳ Mean Squared Error

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

---

## ۵.۱۰ آموزش

### ۵.۱۰.۱ Optimizer

**Adam** با پارامترهای پیش‌فرض:
- Learning Rate: $0.001$
- $\beta_1 = 0.9$, $\beta_2 = 0.999$

```python
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### ۵.۱۰.۲ Learning Rate Scheduler

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
```

### ۵.۱۰.۳ Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### ۵.۱۰.۴ نمونه‌گیری

| پارامتر | مقدار |
|---------|-------|
| Cutoff years | 2017–2022 |
| Max users/persona | 500 |
| Min events/user | 5 |
| Batch size | 256 |
| Train/Val split | 80/20 |

---

## ۵.۱۱ ارزیابی Multi-Year

### ۵.۱۱.۱ روش ارزیابی

```python
# lstm_eval_multi_year.py
for year in [2016, 2017, 2018, 2019, 2020, 2021, 2022]:
    cutoff = datetime(year, 1, 1)
    predictions = model.predict(...)
    metrics = evaluate(predictions, actual_labels)
```

### ۵.۱۱.۲ نتایج

بر اساس فایل `lstm_eval_2016_to_2023.csv`:

| سال | Median F1 (LSTM) | نرخ Churn واقعی |
|-----|------------------|----------------|
| 2016 | 0.60 | ~۲۵٪ |
| 2017 | 0.67 | ~۳۵٪ |
| 2018 | 0.72 | ~۴۵٪ |
| 2019 | 0.78 | ~۵۰٪ |
| 2020 | 0.82 | ~۶۰٪ |
| 2021 | 0.86 | ~۶۵٪ |
| 2022 | 0.88 | ~۷۵٪ |
| 2023 | 0.90 | ~۸۵٪ |

**مشاهده‌ی کلیدی:** F1 مدل LSTM با گذشت زمان **بهتر** می‌شود. دلایل:
- افزایش نرخ Churn واقعی → الگوهای Churn واضح‌تر
- داده‌های آموزشی بیشتر و غنی‌تر
- Era-aware features کمک می‌کند

---

## ۵.۱۲ استخراج Ap/Tp از LSTM

### ۵.۱۲.۱ روش

مدل LSTM علاوه بر Churn، مقادیر $A_p$ و $T_p$ را هم پیش‌بینی می‌کند. این مقادیر می‌توانند:
- برای ارزیابی سیستم Rule-based استفاده شوند
- به‌عنوان پارامترهای dynamic عمل کنند

```python
# extract_ap_tp_from_lstm.py
for persona in range(20):
    predictions = model.predict(persona_users)
    Ap_lstm = predictions['ap'].mean()
    Tp_lstm = predictions['tp'].mean() * 86400  # تبدیل به ثانیه
```

### ۵.۱۲.۲ خروجی

فایل `lstm_derived_ap_tp_2016.csv`:

| persona | Ap | Tp_sec |
|---------|-----|--------|
| 0 | 0.42 | 5,184,000 |
| 1 | 0.38 | 7,776,000 |
| ... | ... | ... |

---

## ۵.۱۳ فایل‌های مرتبط

| فایل | نقش |
|------|-----|
| `lstm_temporal_ap_tp.py` | آموزش مدل |
| `lstm_eval_multi_year.py` | ارزیابی چندساله |
| `extract_ap_tp_from_lstm.py` | استخراج پارامترها |
| `best_temporal_lstm_model.pt` | وزن‌های مدل |
| `lstm_model_technical_documentation.md` | مستندات فنی |

---

## ۵.۱۴ خلاصه‌ی فصل

در این فصل:

- ✅ معماری LSTM چندوظیفه‌ای (~165K پارامتر) توصیف شد
- ✅ سه ورودی: Event Sequence، Global Features، Persona Features
- ✅ سه خروجی: Churn Probability، Ap، Tp
- ✅ Multi-Task Loss با وزن‌های متفاوت
- ✅ Era-aware Features برای مقابله با Temporal Drift
- ✅ بهبود ۱۰–۳۰٪ در F1 نسبت به Rule-based

**نتیجه‌گیری:** LSTM با توجه به توالی رویدادها و Context زمانی، عملکرد بهتری نسبت به Rule-based دارد، به‌خصوص در دوره‌های با Drift بالا.

---

**پایان فصل ۵**

</div>
