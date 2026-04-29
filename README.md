# 🫀 Cardiac AI — Kalp Hastalığı Analiz Sistemi

PyTorch tabanlı 6 katmanlı derin öğrenme modeli ile kalp hastalığı riski tahmini.

## 📁 Dosya Yapısı

```
medical-ai-dashboard/
│
├── app.py                  ← Flask backend + model inference motoru
├── requirements.txt        ← Bağımlılıklar (sadece Flask)
├── README.md               ← Bu dosya
│
└── templates/
    └── index.html          ← Tek sayfalık web arayüzü
```

## 🚀 Kurulum ve Çalıştırma

### 1. Sanal ortam oluşturun (önerilir)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# veya
venv\Scripts\activate           # Windows
```

### 2. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

### 3. Model dosyasını yerleştirin
`best_medicaldataset_pytorch_model_expanded__1_.pth` dosyasını şu konuma koyun:
```
/mnt/user-data/uploads/best_medicaldataset_pytorch_model_expanded__1_.pth
```
**VEYA** `app.py` içindeki `pth_path` değişkenini kendi yolunuzla güncelleyin:
```python
pth_path = 'best_medicaldataset_pytorch_model_expanded__1_.pth'  # kendi yolunuz
```

### 4. Sunucuyu başlatın
```bash
python app.py
```

### 5. Tarayıcıda açın
```
http://localhost:5000
```

---

## 🧠 Model Bilgisi

| Parametre | Değer |
|-----------|-------|
| Model Adı | M10_relu_rmsprop_6layer_800ep |
| Gizli Katmanlar | [256, 128, 96, 64, 32, 16] |
| Aktivasyon | ReLU |
| Optimizer | RMSprop |
| Learning Rate | 0.0003 |
| Dropout | 0.25 |
| Batch Norm | Evet |
| Epoch | 800 |
| **Doğruluk** | **%96.97** |
| **ROC-AUC** | **0.9964** |
| **F1 Skoru** | **0.9552** |
| Karar Eşiği | 0.5266 |

## 📊 Girdi Özellikleri (13 Parametre)

Cleveland Kalp Hastalığı veri setinden:

| Özellik | Açıklama | Aralık |
|---------|----------|--------|
| age | Yaş (yıl) | 20–90 |
| sex | Cinsiyet | 0=Kadın, 1=Erkek |
| cp | Göğüs ağrısı tipi | 0–3 |
| trestbps | İstirahat kan basıncı (mmHg) | 80–220 |
| chol | Serum kolesterol (mg/dl) | 100–600 |
| fbs | Açlık kan şekeri >120 mg/dl | 0/1 |
| restecg | İstirahat EKG | 0–2 |
| thalach | Maks. kalp hızı (bpm) | 60–220 |
| exang | Egzersiz kaynaklı angina | 0/1 |
| oldpeak | ST depresyonu | 0–7 |
| slope | Tepe egzersiz ST eğimi | 0–2 |
| ca | Boyanan büyük damar sayısı | 0–3 |
| thal | Talyum testi | 1–3 |

## ⚙️ Teknik Detaylar

- **Torch bağımlılığı yok**: Model ağırlıkları `.pth` dosyasından saf Python ile okunur (zipfile + struct)
- **Saf Python inference**: `numpy` veya `torch` gerektirmez; tüm matris işlemleri saf Python listelerle yapılır
- **BatchNorm inference modu**: `running_mean` ve `running_var` kullanılır
- **Normalizasyon**: Cleveland veri setinin istatistiksel parametreleriyle StandardScaler simülasyonu

## ⚠️ Uyarı

Bu sistem yalnızca **araştırma ve eğitim amaçlıdır**. Klinik karar desteği, tanı veya tedavi amaçlı kullanılamaz.# do-um-riski-analizi
