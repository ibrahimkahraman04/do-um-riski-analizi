import streamlit as st
from app import predict_with_model, MODEL_INFO, FEATURE_NAMES

st.set_page_config(
    page_title="Maternal Risk AI",
    page_icon="🤰",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: transparent !important;
}

.stApp {
    background: transparent !important;
}

/* ── Animated canvas background ── */
#bg-canvas {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    z-index: -1;
    background: linear-gradient(135deg, #0a0015 0%, #0d0025 40%, #080018 100%);
}

/* ── Mouse glow orb ── */
#mouse-glow {
    position: fixed;
    width: 500px;
    height: 500px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(180,50,255,0.18) 0%, rgba(100,20,200,0.07) 40%, transparent 70%);
    pointer-events: none;
    transform: translate(-50%, -50%);
    transition: left 0.08s ease, top 0.08s ease;
    z-index: 0;
}

/* ── Floating orbs ── */
.orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    pointer-events: none;
    z-index: 0;
    animation: float 8s ease-in-out infinite;
}
.orb1 { width:350px; height:350px; background:rgba(147,51,234,0.25); top:-80px; left:-80px; animation-delay:0s; }
.orb2 { width:300px; height:300px; background:rgba(236,72,153,0.2); bottom:-60px; right:-60px; animation-delay:-3s; }
.orb3 { width:200px; height:200px; background:rgba(59,130,246,0.15); top:40%; left:60%; animation-delay:-5s; }

@keyframes float {
    0%,100% { transform: translateY(0px) scale(1); }
    50% { transform: translateY(-30px) scale(1.05); }
}

/* ── Main content wrapper ── */
.main-wrapper {
    position: relative;
    z-index: 1;
}

/* ── Hero header ── */
.hero-header {
    text-align: center;
    padding: 40px 20px 20px;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(147,51,234,0.2);
    border: 1px solid rgba(147,51,234,0.4);
    border-radius: 50px;
    padding: 6px 18px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #c084fc;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #f8f8ff 0%, #c084fc 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 14px;
}

.hero-sub {
    color: rgba(200,200,220,0.7);
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 30px;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 36px;
}

.stat-chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 10px 20px;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}

.stat-chip:hover {
    background: rgba(147,51,234,0.15);
    border-color: rgba(147,51,234,0.4);
    transform: translateY(-3px);
}

.stat-value { font-size: 1.3rem; font-weight: 700; color: #c084fc; }
.stat-label { font-size: 0.72rem; color: rgba(200,200,220,0.6); text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

/* ── Glass form card ── */
div[data-testid="stForm"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 24px !important;
    padding: 36px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    box-shadow: 0 8px 40px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    transition: box-shadow 0.3s ease !important;
}

div[data-testid="stForm"]:hover {
    box-shadow: 0 12px 60px rgba(147,51,234,0.2), inset 0 1px 0 rgba(255,255,255,0.1) !important;
}

/* ── Labels & text ── */
.stSlider label, .stNumberInput label, label[data-testid="stWidgetLabel"] p {
    color: rgba(220,220,240,0.9) !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, #9333ea, #ec4899) !important;
}

/* ── Submit button ── */
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #9333ea 0%, #ec4899 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.3s cubic-bezier(0.25,0.8,0.25,1) !important;
    box-shadow: 0 4px 20px rgba(147,51,234,0.5) !important;
}

div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(147,51,234,0.7) !important;
}

div[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(1px) !important;
}

/* ── Result cards ── */
.result-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 28px;
    backdrop-filter: blur(20px);
    margin: 12px 0;
    transition: all 0.3s ease;
}

.result-card.high {
    border-color: rgba(239,68,68,0.5);
    background: rgba(239,68,68,0.07);
    box-shadow: 0 0 30px rgba(239,68,68,0.15);
}

.result-card.low {
    border-color: rgba(34,197,94,0.5);
    background: rgba(34,197,94,0.07);
    box-shadow: 0 0 30px rgba(34,197,94,0.15);
}

.result-label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1.5px; color: rgba(200,200,220,0.6); margin-bottom: 6px; }
.result-value { font-size: 2.4rem; font-weight: 800; color: #f8f8ff; }
.result-sublabel { font-size: 0.85rem; color: rgba(200,200,220,0.7); margin-top: 4px; }

/* ── Progress ring ── */
.progress-wrap { display: flex; align-items: center; gap: 16px; margin: 16px 0; }
.risk-bar-bg { flex: 1; height: 10px; background: rgba(255,255,255,0.08); border-radius: 10px; overflow: hidden; }
.risk-bar-fill { height: 100%; border-radius: 10px; transition: width 1s ease; }
.risk-bar-fill.high { background: linear-gradient(90deg,#ef4444,#f97316); }
.risk-bar-fill.medium { background: linear-gradient(90deg,#f59e0b,#eab308); }
.risk-bar-fill.low { background: linear-gradient(90deg,#22c55e,#10b981); }

/* ── Alert overrides ── */
div[data-testid="stAlert"] {
    border-radius: 16px !important;
    border: none !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Metric ── */
div[data-testid="stMetricValue"] { color: #c084fc !important; font-weight: 800 !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ── Section heading ── */
.section-heading {
    font-size: 1.1rem;
    font-weight: 700;
    color: rgba(240,240,255,0.9);
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-sub {
    font-size: 0.82rem;
    color: rgba(180,180,210,0.6);
    margin-bottom: 20px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 30px 0 20px;
    color: rgba(180,180,210,0.4);
    font-size: 0.78rem;
}
</style>

<!-- Background elements -->
<div id="bg-canvas"></div>
<div id="mouse-glow"></div>
<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>

<!-- Hero Header -->
<div class="hero-header">
    <div class="hero-badge">🤖 &nbsp; AI Destekli Analiz</div>
    <div class="hero-title">Maternal Risk<br>Analiz Sistemi</div>
    <div class="hero-sub">PyTorch tabanlı 6 katmanlı derin öğrenme ile doğum riski tahmini</div>
</div>

<!-- Stats Bar -->
<div class="stats-bar">
    <div class="stat-chip">
        <div class="stat-value">%96.97</div>
        <div class="stat-label">Doğruluk</div>
    </div>
    <div class="stat-chip">
        <div class="stat-value">0.9964</div>
        <div class="stat-label">ROC-AUC</div>
    </div>
    <div class="stat-chip">
        <div class="stat-value">0.9552</div>
        <div class="stat-label">F1 Skoru</div>
    </div>
    <div class="stat-chip">
        <div class="stat-value">800</div>
        <div class="stat-label">Epoch</div>
    </div>
</div>

<script>
(function() {
    const glow = document.getElementById('mouse-glow');
    if (!glow) return;
    document.addEventListener('mousemove', function(e) {
        glow.style.left = e.clientX + 'px';
        glow.style.top  = e.clientY + 'px';
    });
})();
</script>
""", unsafe_allow_html=True)

# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("risk_form"):
    st.markdown('<div class="section-heading">📋 Hasta Medikal Verileri</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Aşağıdaki parametreleri hastanın değerlerine göre ayarlayın</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        age           = st.slider("👤 Yaş",                       min_value=10,  max_value=80,  value=25,  step=1,   help="Hastanın yaşı (yıl)")
        body_temp     = st.slider("🌡️ Vücut Sıcaklığı (°F)",     min_value=95.0, max_value=105.0, value=98.6, step=0.1, help="Normal: ~98.6°F")
        heart_rate    = st.slider("💓 Kalp Atış Hızı (bpm)",      min_value=50,  max_value=150, value=75,  step=1,   help="Dakikadaki kalp atışı")
        bmi           = st.slider("⚖️ BMI",                        min_value=15.0, max_value=50.0, value=25.0, step=0.1, help="Vücut Kitle İndeksi")

    with col2:
        systolic_bp   = st.slider("🩺 Sistolik KB (mmHg)",        min_value=80,  max_value=180, value=120, step=1,   help="Büyük tansiyon")
        diastolic_bp  = st.slider("🩺 Diastolik KB (mmHg)",       min_value=50,  max_value=120, value=80,  step=1,   help="Küçük tansiyon")
        hba1c         = st.slider("🩸 HbA1c (%)",                  min_value=4.0, max_value=12.0, value=5.5, step=0.1, help="Son 3 ay kan şekeri ortalaması")
        fasting_glucose = st.slider("🍬 Açlık Kan Şekeri (mg/dL)", min_value=60,  max_value=200, value=90,  step=1,   help="8-12 saat açlık sonrası")

    st.write("")
    submit = st.form_submit_button("🚀  Analizi Başlat")

# ── Result ────────────────────────────────────────────────────────────────────
if submit:
    features = [
        float(age), float(body_temp), float(heart_rate),
        float(systolic_bp), float(diastolic_bp), float(bmi),
        float(hba1c), float(fasting_glucose)
    ]

    try:
        prob, pred = predict_with_model(features)
        pct = round(prob * 100, 2)

        if prob >= 0.7:
            level, level_label, bar_class = "HIGH", "YÜKSEK RİSK", "high"
        elif prob >= 0.4:
            level, level_label, bar_class = "MEDIUM", "ORTA RİSK", "medium"
        else:
            level, level_label, bar_class = "LOW", "DÜŞÜK RİSK", "low"

        card_class = "high" if pred == 1 else "low"
        icon = "⚠️" if pred == 1 else "✅"
        verdict = "Yüksek Doğum Riski Tespit Edildi" if pred == 1 else "Normal / Düşük Risk Seviyesi"

        st.markdown("---")
        st.markdown('<div class="section-heading">📊 Analiz Sonucu</div>', unsafe_allow_html=True)
        st.write("")

        r1, r2 = st.columns(2, gap="large")

        with r1:
            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="result-label">Risk Olasılığı</div>
                <div class="result-value">%{pct}</div>
                <div class="result-sublabel">Karar Eşiği: %{round(MODEL_INFO['best_threshold']*100,1)}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="result-card {card_class}">
                <div class="result-label">Tanı</div>
                <div class="result-value" style="font-size:1.4rem;">{icon} {level_label}</div>
                <div class="result-sublabel">{verdict}</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown(f"""
        <div class="progress-wrap">
            <span style="color:rgba(200,200,220,0.6);font-size:0.8rem;min-width:80px;">Risk Seviyesi</span>
            <div class="risk-bar-bg">
                <div class="risk-bar-fill {bar_class}" style="width:{min(pct,100)}%;"></div>
            </div>
            <span style="color:#c084fc;font-weight:700;font-size:0.9rem;min-width:50px;">%{pct}</span>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        if level == "HIGH":
            st.error("⚠️ **Yüksek Risk** — Değerler kritik seviyede. Lütfen derhal bir uzman doktora başvurun.")
        elif level == "MEDIUM":
            st.warning("🔶 **Orta Risk** — Değerler yakından izlenmeli ve bir hekim kontrolü önerilir.")
        else:
            st.success("✅ **Düşük Risk** — Değerler normal aralıkta. Düzenli takip önerilir.")

    except Exception as e:
        st.error(f"Hata: {e}")

# Footer
st.markdown("""
<div class="footer">
    ⚠️ Bu sistem yalnızca <b>araştırma amaçlıdır</b>. Klinik tanı veya tedavi amacıyla kullanılamaz.<br>
    Maternal AI · M10_relu_rmsprop_6layer_800ep · 6 Katman · 800 Epoch
</div>
""", unsafe_allow_html=True)
