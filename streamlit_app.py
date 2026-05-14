import streamlit as st
from app import predict_with_model, MODEL_INFO, FEATURE_NAMES

# Sayfa Ayarları
st.set_page_config(
    page_title="Maternal Risk Analizi",
    page_icon="🤰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Özel CSS ile Modern Tasarım
st.markdown("""
<style>
    /* Genel Arka Plan ve Yazı Tipi */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Form Kartı Tasarımı */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s ease-in-out;
    }
    
    div[data-testid="stForm"]:hover {
        box-shadow: 0 10px 40px 0 rgba(31, 38, 135, 0.25);
    }
    
    /* Buton Tasarımı */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: 600;
        letter-spacing: 1px;
        width: 100%;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    
    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.6);
    }
    
    div[data-testid="stFormSubmitButton"] > button:active {
        transform: translateY(1px);
    }
    
    /* Başlıklar ve Metinler */
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #FF8E53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        margin-bottom: 0.5rem !important;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    
    /* Başarı ve Hata Mesajları */
    div[data-testid="stAlert"] {
        border-radius: 15px;
        border: none;
    }
    
    /* Metrik Değerleri */
    div[data-testid="stMetricValue"] {
        font-weight: 700 !important;
        color: #FF6B6B !important;
    }
</style>
""", unsafe_allow_html=True)

# Başlık ve Açıklama
st.markdown("<h1>🤰 Doğum Riski Analizi</h1>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>Makine öğrenimi ile maternal risk tahmini. (Model Doğruluğu: <b>%{MODEL_INFO['accuracy']*100:.2f}</b>)</div>", unsafe_allow_html=True)

# Form oluşturma
with st.form("risk_form"):
    st.markdown("### 📋 Medikal Veriler")
    st.markdown("Lütfen risk analizi için aşağıdaki değerleri doldurun.")
    st.write("") # Boşluk
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Yaş", min_value=10, max_value=80, value=25, step=1, help="Hastanın yaşı")
        body_temp = st.slider("Vücut Sıcaklığı (°F)", min_value=95.0, max_value=105.0, value=98.6, step=0.1, help="Normal değer yaklaşık 98.6°F")
        heart_rate = st.slider("Kalp Atış Hızı (bpm)", min_value=50, max_value=150, value=75, step=1, help="Dakikadaki kalp atış hızı")
        bmi = st.slider("Vücut Kitle İndeksi (BMI)", min_value=15.0, max_value=50.0, value=25.0, step=0.1, help="Kilo(kg) / Boy(m)²")
        
    with col2:
        systolic_bp = st.slider("Sistolik Kan Basıncı", min_value=80, max_value=180, value=120, step=1, help="Büyük tansiyon")
        diastolic_bp = st.slider("Diastolik Kan Basıncı", min_value=50, max_value=120, value=80, step=1, help="Küçük tansiyon")
        hba1c = st.slider("HbA1c Seviyesi (%)", min_value=4.0, max_value=12.0, value=5.5, step=0.1, help="Son 3 aylık kan şekeri ortalaması")
        fasting_glucose = st.slider("Açlık Kan Şekeri (mg/dL)", min_value=60, max_value=200, value=90, step=1, help="8-12 saatlik açlık sonrası ölçülen şeker")

    st.write("") # Boşluk
    submit_button = st.form_submit_button("🚀 Analizi Başlat")

if submit_button:
    # app.py'nin beklediği sırayla özellikleri listeye koyuyoruz
    features = [
        float(age),
        float(body_temp),
        float(heart_rate),
        float(systolic_bp),
        float(diastolic_bp),
        float(bmi),
        float(hba1c),
        float(fasting_glucose)
    ]
    
    try:
        prob, pred = predict_with_model(features)
        
        st.write("")
        st.markdown("---")
        st.markdown("<h2 style='text-align: center;'>📊 Analiz Sonucu</h2>", unsafe_allow_html=True)
        
        # Sonuç Gösterimi için Kolonlar
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.metric(label="Risk Olasılığı", value=f"%{round(prob * 100, 2)}")
            
        with res_col2:
            if pred == 1:
                st.error("⚠️ **Yüksek Doğum Riski Tespit Edildi**")
            else:
                st.success("✅ **Normal / Düşük Risk Seviyesi**")
        
        # Ekstra Bilgi ve İlerlemeler
        st.write("")
        st.progress(min(prob, 1.0))
        
        if prob >= 0.7:
            st.warning("Risk Seviyesi: **YÜKSEK** - Değerler kritik seviyede olabilir. Lütfen bir uzman doktora başvurun.")
        elif prob >= 0.4:
            st.info("Risk Seviyesi: **ORTA** - Değerler yakından izlenmelidir.")
        else:
            st.info("Risk Seviyesi: **DÜŞÜK** - Değerler normal görünmektedir.")
            
    except Exception as e:
        st.error(f"Tahmin işlemi sırasında bir hata oluştu: {e}")
