import streamlit as st
from app import predict_with_model, MODEL_INFO, FEATURE_NAMES

# Sayfa Ayarları
st.set_page_config(
    page_title="Maternal Risk Analizi",
    page_icon="🤰",
    layout="centered"
)

# Başlık ve Açıklama
st.title("🤰 Doğum Riski Analizi (Maternal Risk)")
st.markdown(f"**Model:** {MODEL_INFO['name']} | **Doğruluk:** %{MODEL_INFO['accuracy']*100:.2f}")
st.write("Lütfen hastanın medikal verilerini girerek risk analizini başlatın.")

# Form oluşturma
with st.form("risk_form"):
    st.subheader("Hasta Medikal Verileri")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Yaş", min_value=10.0, max_value=100.0, value=25.0, step=1.0)
        body_temp = st.number_input("Vücut Sıcaklığı (°F)", min_value=90.0, max_value=110.0, value=98.6, step=0.1)
        heart_rate = st.number_input("Kalp Atış Hızı (bpm)", min_value=40.0, max_value=200.0, value=75.0, step=1.0)
        bmi = st.number_input("Vücut Kitle İndeksi (BMI)", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        
    with col2:
        systolic_bp = st.number_input("Sistolik Kan Basıncı (Büyük Tansiyon)", min_value=70.0, max_value=250.0, value=120.0, step=1.0)
        diastolic_bp = st.number_input("Diastolik Kan Basıncı (Küçük Tansiyon)", min_value=40.0, max_value=150.0, value=80.0, step=1.0)
        hba1c = st.number_input("HbA1c Seviyesi (%)", min_value=3.0, max_value=20.0, value=5.5, step=0.1)
        fasting_glucose = st.number_input("Açlık Kan Şekeri (mg/dL)", min_value=40.0, max_value=400.0, value=90.0, step=1.0)

    submit_button = st.form_submit_button("Analiz Et")

if submit_button:
    # app.py'nin beklediği sırayla özellikleri listeye koyuyoruz
    # Sıra: "age", "body_temp", "heart_rate", "systolic_bp", "diastolic_bp", "bmi", "hba1c", "fasting_glucose"
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
        # app.py içindeki hazır tahmin fonksiyonunu çağırıyoruz
        prob, pred = predict_with_model(features)
        
        st.divider()
        st.subheader("Analiz Sonucu")
        
        if pred == 1:
            st.error("⚠️ Yüksek Doğum Riski Tespit Edildi")
        else:
            st.success("✅ Normal / Düşük Risk Seviyesi")
            
        st.metric(label="Risk Olasılığı", value=f"%{round(prob * 100, 2)}")
        
        if prob >= 0.7:
            st.warning("Risk Seviyesi: YÜKSEK (Lütfen bir uzman doktora başvurun.)")
        elif prob >= 0.4:
            st.info("Risk Seviyesi: ORTA")
        else:
            st.info("Risk Seviyesi: DÜŞÜK")
            
    except Exception as e:
        st.error(f"Tahmin işlemi sırasında bir hata oluştu: {e}")
