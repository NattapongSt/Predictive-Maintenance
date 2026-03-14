# Predictive Maintenance — Vibration Analysis
> วิเคราะห์การสั่นสะเทือนของเครื่องจักรด้วย Deep Learning เพื่อพยากรณ์ความผิดปกติล่วงหน้า  
> มาตรฐาน ISO 10816-3 | CNN vs LSTM

---

## โจทย์คืออะไร?

เครื่องจักรในโรงงานถ้าพังกะทันหันจะหยุดสายการผลิตและสูญเสียเงินมหาศาล  
โปรเจกต์นี้ใช้ **เซ็นเซอร์วัดการสั่นสะเทือน** + **AI** เพื่อรู้ล่วงหน้าว่าเครื่องจักรอยู่ในสภาพไหน ก่อนที่มันจะพัง

| วิธีเดิม | วิธีใหม่ (Predictive Maintenance) |
|---|---|
| รอให้พังแล้วค่อยซ่อม | รู้ล่วงหน้าว่าจะพัง |
| ซ่อมตามกำหนด ทั้งที่อาจยังดีอยู่ | ซ่อมเฉพาะเมื่อจำเป็น |
| ช่างต้องเดินตรวจทุกวัน | ระบบแจ้งเตือนอัตโนมัติ |

---

## เครื่องจักรที่ติดตาม

- Motor Compressor OAH-06_A
- (CHPP) Motor Compressor CH-06_A
- Cooling Pump for OAH-02
- (CHPP) Cooling Pump for OAH-02 / ECH-02
- Jockey Pump

---

## ข้อมูลดิบ (ml_dataset.csv)

เซ็นเซอร์บันทึกค่าการสั่นสะเทือนทุกๆ ~133 ms ต่อการวัด 1 ครั้ง

| คอลัมน์ | คำอธิบาย | หน่วย |
|---|---|---|
| `Equipment` | ชื่อเครื่องจักร | — |
| `Meas. Point` | จุดติดตั้งเซ็นเซอร์ | — |
| `Datetime` | วันเวลาที่วัด | timestamp |
| `Time (mS)` | เวลาภายใน session | milliseconds |
| `Amplitude` | ความแรงของการสั่น | g (แรงโน้มถ่วง) |

```
ขนาด : 51,186 rows × 5 columns
เครื่อง : 6 เครื่อง
Sessions : 11 ครั้งวัด (แต่ละครั้ง ~2,048–20,475 จุด)
```

---

## Pipeline ทีละขั้นตอน

### ขั้นที่ 1 — โหลดและสำรวจข้อมูล

```python
df = pd.read_csv('ml_dataset.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%b-%y %H:%M:%S')
```

ตรวจสอบจำนวนเครื่อง, sessions, และ RMS เบื้องต้นของแต่ละเครื่อง

---

### ขั้นที่ 2 — จัดกลุ่มเป็น Session

**Session** = ข้อมูลการสั่นสะเทือน 1 ชุด จาก 1 เครื่อง ใน 1 วันเวลา

```python
for (equipment, datetime), group in df.groupby(['Equipment', 'Datetime']):
    # group คือสัญญาณทั้งหมดของการวัดครั้งนั้น
    ...
```

> ข้อมูล 51,186 แถวรวมอยู่ในไฟล์เดียว ต้องแยกออกก่อนเพราะแต่ละครั้งคือสัญญาณคนละชุดกัน

---

### ขั้นที่ 3 — FFT (Fast Fourier Transform)

แปลงสัญญาณจาก **time-domain** → **frequency-domain**

```python
def compute_fft(time_ms, amplitude):
    window    = np.hanning(N)           # Hann window ลด spectral leakage
    fft_vals  = fft(amplitude * window)
    freqs     = fftfreq(N, d=dt)
    # คืนค่าเฉพาะส่วนความถี่บวก (one-sided)
    return freqs[freqs >= 0], magnitude[freqs >= 0]
```

| | ก่อน FFT | หลัง FFT |
|---|---|---|
| แกน X | เวลา (ms) | ความถี่ (Hz) |
| แกน Y | amplitude ขึ้น-ลง | ขนาดของแต่ละความถี่ |
| บอกอะไร | "ตอนนี้สั่นแรงแค่ไหน" | "สั่นที่ความถี่ไหนมากที่สุด" |

> **ทำไมต้องทำ FFT?**  
> ความผิดปกติแต่ละประเภทสั่นที่ความถี่ต่างกัน เช่น bearing เสียจะส่งพลังงานที่ 200–1,000 Hz  
> ถ้าดูแค่สัญญาณดิบจะเห็นแค่ "สั่นแรง" แต่ไม่รู้ว่าปัญหาคืออะไร

---

### ขั้นที่ 4 — Feature Extraction

สรุปสัญญาณ ~4,000 จุดต่อ session ให้เหลือ **15 ตัวเลข** ที่อธิบายลักษณะของสัญญาณ

#### 9 features จาก Time-domain

| Feature | สูตร | ตีความ |
|---|---|---|
| `rms` | √(mean(x²)) | พลังงานเฉลี่ย ความรุนแรงโดยรวม |
| `peak` | max(\|x\|) | จุดสูงสุด worst case |
| `peak_to_peak` | max(x) − min(x) | ช่วงการแกว่ง |
| `crest_factor` | peak / rms | ถ้าสูง = มี spike กะทันหัน |
| `kurtosis` | 4th moment / σ⁴ | ถ้า > 3 = มี impulse บ่อย → bearing อาจเสีย |
| `skewness` | 3rd moment / σ³ | ความเบ้ของสัญญาณ |
| `std` | √(mean((x−μ)²)) | ความแปรปรวน |
| `shape_factor` | rms / mean(\|x\|) | รูปร่างของคลื่น |
| `impulse_factor` | peak / mean(\|x\|) | ความรุนแรงของการกระแทก |

#### 6 features จาก Frequency-domain (หลัง FFT)

| Feature | ตีความ |
|---|---|
| `dominant_freq` | ความถี่ที่โดดเด่นที่สุด |
| `spectral_centroid` | "จุดศูนย์กลาง" ของสเปกตรัม |
| `band_energy_1` | % พลังงานที่ 0–50 Hz (การหมุนพื้นฐาน, imbalance) |
| `band_energy_2` | % พลังงานที่ 50–200 Hz (misalignment) |
| `band_energy_3` | % พลังงานที่ 200–1,000 Hz (bearing fault) |
| `band_energy_4` | % พลังงานที่ > 1,000 Hz (high-frequency noise) |

---

### ขั้นที่ 5 — ISO 10816-3 Zone Classification (Label)

แปลง RMS amplitude (g) → velocity (mm/s) แล้วจัดโซน

```python
def accel_to_velocity(rms_g, freq_hz):
    return (rms_g * 9.81) / (2 * π * freq_hz) * 1000  # mm/s

def classify_iso10816(velocity_mm_s):
    if   velocity < 2.3:  return 0, 'Zone A'  # ดีเยี่ยม
    elif velocity < 4.5:  return 1, 'Zone B'  # ปกติ
    elif velocity < 7.1:  return 2, 'Zone C'  # แจ้งเตือน
    else:                 return 3, 'Zone D'  # อันตราย — หยุดเครื่องทันที
```

| Zone | ความเร็วสั่น | ความหมาย | สิ่งที่ต้องทำ |
|---|---|---|---|
| **Zone A** | < 2.3 mm/s | ดีเยี่ยม | ไม่ต้องทำอะไร |
| **Zone B** | 2.3–4.5 mm/s | ปกติ | ติดตามต่อเนื่อง |
| **Zone C** | 4.5–7.1 mm/s | แจ้งเตือน | วางแผนซ่อม |
| **Zone D** | > 7.1 mm/s | อันตราย | หยุดเดินเครื่องทันที |

#### ผลสำหรับข้อมูลชุดนี้

| เครื่องจักร | Velocity | Zone |
|---|---|---|
| Motor Compressor OAH-06_A | 0.15–0.17 mm/s | 🟢 Zone A |
| Cooling Pump | 0.74–0.89 mm/s | 🟢 Zone A |
| Jockey Pump (ครั้งที่ 1) | 9.25 mm/s | 🔴 Zone D |
| Jockey Pump (ครั้งที่ 2–3) | 5.0–5.1 mm/s | 🟡 Zone C |

---

### ขั้นที่ 6 — Data Augmentation + Normalize + Split

#### ทำไมต้อง Augment?
มีข้อมูลจริงแค่ 11 sessions ซึ่งน้อยเกินไปสำหรับ train model  
จึงสร้างข้อมูลจำลองโดยเพิ่ม noise และ scaling เล็กน้อย เพื่อจำลองความแปรปรวนที่เกิดขึ้นได้จริงในการวัด

```python
for i in range(len(X)):
    noise  = np.random.normal(0, 0.03, n_features)  # noise เล็กน้อย
    scale  = np.random.uniform(0.95, 1.05)           # ปรับขนาด ±5%
    X_aug.append(X[i] * scale + noise)
# 11 sessions → 300 samples
```

#### Normalize ด้วย StandardScaler
Features 15 ตัวมีขนาดต่างกันมาก เช่น RMS ~0.5 แต่ spectral centroid อาจเป็น 500  
StandardScaler แปลงทุก feature ให้ mean=0, std=1 เพื่อให้แต่ละ feature มีน้ำหนักเท่ากัน

```python
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_aug)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_aug, test_size=0.2, random_state=42, stratify=y_aug
)
# Train: 240 samples | Test: 60 samples
```

---

### ขั้นที่ 7 — CNN Model

รับ **15 features** ของ 1 session → ทำนาย Zone

```
Input  : [15 features]
Dense 256  ReLU          ← จำลอง Conv + MaxPool
Dense 128  ReLU          ← จำลอง Conv + MaxPool
Dense  64  ReLU          ← Fully connected
Dense   4  Softmax
Output : [Zone A, Zone B, Zone C, Zone D]
```

```python
cnn_model = MLPClassifier(
    hidden_layer_sizes  = (256, 128, 64),
    activation          = 'relu',
    solver              = 'adam',
    learning_rate_init  = 0.001,
    early_stopping      = True,
)
```

**เหมาะกับ:** ต้องการผลทันทีทุกครั้งที่วัด, เครื่องมีประวัติข้อมูลน้อย, real-time alert

---

### ขั้นที่ 8 — LSTM Model

รับ **5 sessions ติดกัน** (sliding window) → ทำนาย Zone  
มองแนวโน้มตามเวลา ว่าเครื่อง "ค่อยๆ แย่ลง" ไหม

```
Input  : [5 steps × 15 features = 75]
Dense 512  tanh           ← จำลอง LSTM layer 1
Dense 256  tanh           ← จำลอง LSTM layer 2
Dense  64  ReLU           ← Fully connected
Dense   4  Softmax
Output : [Zone A, Zone B, Zone C, Zone D]
```

```python
# สร้าง sliding window sequences
for i in range(len(X) - seq_len):
    X_seq.append(X[i : i + seq_len].flatten())  # 5 sessions ติดกัน
    y_seq.append(y[i + seq_len])

lstm_model = MLPClassifier(
    hidden_layer_sizes  = (512, 256, 64),
    activation          = 'tanh',   # LSTM-like activation
    learning_rate_init  = 0.0005,
    early_stopping      = True,
)
```

**เหมาะกับ:** มีข้อมูลสะสมหลายเดือน, ต้องการจับแนวโน้มการเสื่อมสภาพ

---

## ผลลัพธ์

| Model | Accuracy | F1 (macro) | จุดเด่น |
|---|---|---|---|
| CNN | **98.3%** | 0.96 | เร็ว ใช้งานได้ทันที |
| LSTM | **98.3%** | 0.96 | จับ trend เสื่อมสภาพได้ดีกว่าระยะยาว |

> Accuracy เท่ากันเพราะข้อมูลจริงมีแค่ 11 sessions ถ้ามีข้อมูลมากกว่านี้ LSTM มักได้เปรียบใน predictive maintenance

---

## เลือก Model ไหนดี?

```
มีข้อมูลน้อย หรือต้องการ real-time alert
    → ใช้ CNN

มีข้อมูลสะสมหลายเดือน
    → ใช้ LSTM

ต้องการทั้งความแม่นยำและการจับ trend (แนะนำ)
    → ใช้ทั้งคู่ร่วมกัน
```

---

## Dependencies

```bash
pip install pandas numpy scipy scikit-learn matplotlib
```

| Library | การใช้งาน |
|---|---|
| `pandas` | โหลดและจัดการข้อมูล |
| `numpy` | คำนวณ features |
| `scipy.fft` | FFT processing |
| `sklearn` | StandardScaler, MLPClassifier, metrics |
| `matplotlib` | สร้างกราฟรายงาน |

---

## อ้างอิงมาตรฐาน

- **ISO 10816-3** — Mechanical vibration: Evaluation of machine vibration by measurements on non-rotating parts  
  ใช้กำหนดเกณฑ์ Zone A/B/C/D สำหรับเครื่องจักรอุตสาหกรรม