import streamlit as st
import pandas as pd
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# ── Language ───────────────────────────────────────────────────────────────────
LANG = {
    "en": {
        "title": "Laptop Price Predictor",
        "subtitle": "",
        "section": "📋 Laptop Specifications",
        "brand": "Brand", "model": "Model", "ram": "RAM (GB)",
        "storage": "Storage (GB)", "storage_type": "Storage Type",
        "cpu": "CPU Generation", "year": "Year", "condition": "Condition",
        "screen": "Screen Size (inch)", "gpu": "GPU", "touch": "Touchscreen",
        "button": "⚡ Predict Price", "est_label": "✨ Estimated Price Range",
        "low": "Low", "mid": "Mid", "high": "High",
        "currency_label": "💱 Currency",
        "live_rates": "✅ Live exchange rates loaded",
        "fallback_rates": "⚠️ Using cached rates",
        "footer": "Powered by Random Forest · Built with Streamlit",
    },
    "ar": {
        "title": "توقع سعر اللاب توب",
        "subtitle": "",
        "section": "📋 مواصفات اللاب توب",
        "brand": "الماركة", "model": "الموديل", "ram": "الرام (جيجا)",
        "storage": "التخزين (جيجا)", "storage_type": "نوع التخزين",
        "cpu": "جيل المعالج", "year": "سنة الصنع", "condition": "الحالة",
        "screen": "حجم الشاشة (إنش)", "gpu": "كارت الشاشة", "touch": "شاشة لمس",
        "button": "⚡ توقع السعر", "est_label": "✨ نطاق السعر المتوقع",
        "low": "الأدنى", "mid": "المتوسط", "high": "الأعلى",
        "currency_label": "💱 العملة",
        "live_rates": "✅ تم تحميل أسعار الصرف الحية",
        "fallback_rates": "⚠️ يتم استخدام أسعار محفوظة",
        "footer": "يعمل بـ Random Forest · مبني بـ Streamlit",
    }
}

FALLBACK_RATES = {
    "🇪🇬 EGP": 1.0, "🇺🇸 USD": 0.01990,
    "🇪🇺 EUR": 0.01710, "🇸🇦 SAR": 0.07450, "🇦🇪 AED": 0.07320,
}

@st.cache_data(ttl=86400)
def fetch_live_rates():
    try:
        res  = requests.get("https://open.er-api.com/v6/latest/EGP", timeout=5)
        data = res.json()
        if data.get("result") == "success":
            r = data["rates"]
            return {"🇪🇬 EGP": 1.0,
                    "🇺🇸 USD": round(r.get("USD", 0.01990), 6),
                    "🇪🇺 EUR": round(r.get("EUR", 0.01710), 6),
                    "🇸🇦 SAR": round(r.get("SAR", 0.07450), 6),
                    "🇦🇪 AED": round(r.get("AED", 0.07320), 6)}, True
    except Exception:
        pass
    return FALLBACK_RATES, False

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Cairo:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    background: linear-gradient(135deg, #fff9f0 0%, #fff0f9 50%, #f0f4ff 100%);
    color: #2d2d3a;
}
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 760px !important; }

/* ── Hero ── */
.hero { text-align: center; padding: 2.5rem 1rem 1.5rem; }
.hero-icon { font-size: 3.5rem; display: block; margin-bottom: 0.4rem; animation: bounce 2s infinite; }
@keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
.hero h1 {
    font-size: 2.6rem; font-weight: 900;
    background: linear-gradient(135deg, #ff6b6b, #ffa500, #ff6bcd);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.15; margin-bottom: 0.5rem;
}

/* ── Live badge ── */
.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0fff4; border: 1.5px solid #68d391;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.75rem; font-weight: 700; color: #276749; letter-spacing: 0.04em;
}
.live-dot { width: 7px; height: 7px; border-radius: 50%; background: #48bb78; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

/* ── Card ── */
.card {
    background: #ffffff;
    border: 2px solid #ffe0f0;
    border-radius: 24px;
    padding: 2rem; margin-bottom: 1.5rem;
    box-shadow: 0 8px 30px #ff6b6b18, 0 2px 8px #ffa50018;
}
.card-title {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 0.1em;
    text-transform: uppercase; color: #ff6bcd; margin-bottom: 1.2rem;
}

/* ── Labels ── */
label, .stSelectbox label {
    font-size: 0.82rem !important; font-weight: 700 !important;
    color: #f97316 !important; letter-spacing: 0.03em !important; text-transform: uppercase !important;
}

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background-color: #fff8f0 !important; border: 2px solid #fcd9b0 !important;
    border-radius: 12px !important; color: #2d2d3a !important;
    font-size: 0.95rem !important; transition: border-color 0.2s;
}
div[data-baseweb="select"] > div:hover { border-color: #ff6b6b !important; }
div[data-baseweb="select"] > div:focus-within { border-color: #ff6bcd !important; box-shadow: 0 0 0 3px #ff6bcd22 !important; }
ul[role="listbox"] { background-color: #fffaf5 !important; border: 2px solid #fcd9b0 !important; border-radius: 12px !important; }
li[role="option"]:hover { background-color: #fff0e0 !important; }

/* ── Button ── */
div.stButton > button {
    width: 100%; padding: 0.9rem 2rem;
    background: linear-gradient(135deg, #ff6b6b, #ff9f43, #ff6bcd);
    background-size: 200% auto;
    color: #fff; font-size: 1.05rem; font-weight: 800; letter-spacing: 0.05em;
    border: none; border-radius: 16px; cursor: pointer;
    transition: all 0.3s ease; box-shadow: 0 6px 20px #ff6b6b44; margin-top: 0.5rem;
}
div.stButton > button:hover {
    transform: translateY(-3px) scale(1.01);
    box-shadow: 0 10px 28px #ff6b6b66;
    background-position: right center;
}

/* ── Result boxes ── */
.range-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-top: 1.2rem; }
.range-box { border-radius: 20px; padding: 1.3rem 0.8rem; text-align: center; animation: fadeUp 0.4s ease; }
.range-box.low  { background: linear-gradient(135deg, #e0f7ff, #b3ecff); border: 2px solid #4fc3f7; }
.range-box.mid  { background: linear-gradient(135deg, #ffe0f5, #ffb3e6); border: 2px solid #f06292; box-shadow: 0 6px 24px #ff6bcd33; }
.range-box.high { background: linear-gradient(135deg, #fff3e0, #ffe0b2); border: 2px solid #ffa726; }
.range-label { font-size: 0.72rem; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.4rem; }
.low  .range-label { color: #0288d1; }
.mid  .range-label { color: #e91e8c; }
.high .range-label { color: #ef6c00; }
.range-price { font-family: 'Nunito', sans-serif; font-size: 1.6rem; font-weight: 900; color: #2d2d3a; }
.range-currency { font-size: 0.72rem; color: #999; margin-top: 0.2rem; font-weight: 600; }
.result-title { font-size: 0.78rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase; color: #ff6bcd; margin-bottom: 0.4rem; text-align: center; }

@keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }

/* ── RTL ── */
.rtl { direction: rtl; text-align: right; font-family: 'Cairo', sans-serif !important; }
.rtl label, .rtl .card-title, .rtl .result-title { font-family: 'Cairo', sans-serif !important; letter-spacing: 0 !important; }
.stRadio > div { flex-direction: row !important; gap: 0.5rem; }
.stRadio label { font-size: 0.85rem !important; color: #ff6b6b !important; font-weight: 700 !important; }
.footer { text-align: center; color: #bbb; font-size: 0.78rem; margin-top: 3rem; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Load & Train ───────────────────────────────────────────────────────────────
CAT_COLS = ["Brand", "Model", "Storage_Type", "Condition", "GPU", "Touchscreen"]

@st.cache_resource
def load_model():
    df = pd.read_excel("hhhhhema.xlsx")
    encoders = {}
    for col in CAT_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    X = df.drop("Price", axis=1)
    y = df["Price"]
    mdl = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    mdl.fit(X, y)
    return mdl, encoders

@st.cache_data
def load_raw():
    return pd.read_excel("hhhhhema.xlsx")

model, encoders = load_model()
df_orig = load_raw()
CURRENCIES, is_live = fetch_live_rates()

# ── Language toggle ────────────────────────────────────────────────────────────
lang_choice = st.radio("", ["🇬🇧 English", "🇪🇬 العربية"], horizontal=True)
lang = "ar" if "العربية" in lang_choice else "en"
L   = LANG[lang]
rtl = "rtl" if lang == "ar" else ""

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero {rtl}">
    <span class="hero-icon">💻</span>
    <h1>{L['title']}</h1>
    <p>{L['subtitle']}</p>
</div>
""", unsafe_allow_html=True)

if is_live:
    st.markdown(f'<div style="text-align:center;margin-bottom:1rem"><span class="live-badge"><span class="live-dot"></span>{L["live_rates"]}</span></div>', unsafe_allow_html=True)
else:
    st.warning(L["fallback_rates"])

# ── Currency ───────────────────────────────────────────────────────────────────
st.markdown(f'<div class="card {rtl}"><div class="card-title">{L["currency_label"]}</div>', unsafe_allow_html=True)
currency = st.selectbox("", list(CURRENCIES.keys()), label_visibility="collapsed")
rate = CURRENCIES[currency]
currency_code = currency.split(" ")[1]
st.markdown('</div>', unsafe_allow_html=True)

# ── Form ───────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="card {rtl}"><div class="card-title">{L["section"]}</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    brand       = st.selectbox(L['brand'], sorted(df_orig["Brand"].unique()))
    models_list = sorted(df_orig[df_orig["Brand"] == brand]["Model"].unique())
    model_sel   = st.selectbox(L['model'], models_list)

    # Filter all options strictly by Brand + Model
    filtered = df_orig[(df_orig["Brand"] == brand) & (df_orig["Model"] == model_sel)]
    f        = filtered if not filtered.empty else df_orig

    ram          = st.selectbox(L['ram'],          sorted(f["RAM"].unique()))
    storage      = st.selectbox(L['storage'],      sorted(f["Storage"].unique()))
    storage_type = st.selectbox(L['storage_type'], sorted(f["Storage_Type"].unique()))

with col2:
    cpu_gen   = st.selectbox(L['cpu'],       sorted(f["CPU_Gen"].unique()))
    # Year — only show from release year up to last known version year
    MODEL_YEAR_RANGE = {
        "Inspiron 15 3520": (2022,2026), "Inspiron 15 5520": (2022,2026),
        "Inspiron 14 5420": (2022,2026), "XPS 13 9315": (2022,2026),
        "XPS 15 9520": (2022,2026), "G15 5520": (2022,2026),
        "Vostro 3520": (2022,2026), "Latitude 5530": (2022,2026),
        "Pavilion 15 eg2": (2022,2026), "Envy x360 15": (2022,2026),
        "Envy x360 13": (2022,2026), "Spectre x360 14": (2022,2026),
        "EliteBook 840 G9": (2022,2026), "Omen 16": (2021,2026),
        "Victus 15": (2021,2026), "HP 250 G9": (2022,2026),
        "IdeaPad 3 15": (2021,2026), "IdeaPad 5 15": (2021,2026),
        "IdeaPad Slim 5": (2022,2026), "ThinkPad E14 Gen4": (2022,2026),
        "ThinkPad X1 Carbon Gen10": (2022,2026), "Yoga 7 14": (2022,2026),
        "Legion 5 Gen7": (2022,2024), "Legion 5 Pro Gen7": (2022,2024),
        "MacBook Air M1": (2020,2023), "MacBook Air M2": (2022,2026),
        "MacBook Air M2 15": (2023,2026), "MacBook Pro 13 M2": (2022,2023),
        "MacBook Pro 14 M2": (2023,2026), "MacBook Pro 16 M2": (2023,2026),
        "VivoBook 15 X1502": (2022,2026), "VivoBook 14 X1402": (2021,2026),
        "ZenBook 14 UX425": (2021,2023), "ZenBook Pro 15": (2022,2026),
        "ROG Strix G15": (2021,2026), "TUF Gaming A15": (2021,2026),
        "TUF Gaming F15": (2021,2026), "ROG Zephyrus G14": (2022,2026),
        "Aspire 5 A515": (2021,2026), "Aspire 3 A315": (2021,2026),
        "Swift 3 SF314": (2022,2026), "Swift X SFX14": (2022,2026),
        "Nitro 5 AN515": (2021,2026), "Predator Helios 300": (2021,2026),
        "GF63 Thin 12V": (2022,2026), "Katana GF66": (2021,2023),
        "Raider GE66": (2021,2023), "GS66 Stealth": (2021,2023),
        "Modern 14 B12M": (2022,2026), "Prestige 14 EVO": (2022,2026),
        "Creator 15": (2021,2026),
        "Galaxy Book2": (2022,2023), "Galaxy Book2 Pro": (2022,2023),
        "Galaxy Book2 Pro 360": (2022,2023), "Galaxy Book3 Pro": (2023,2026),
        "Galaxy Book3 Ultra": (2023,2026),
        "MateBook D14 2022": (2022,2026), "MateBook D15 2022": (2022,2026),
        "MateBook X Pro 2022": (2022,2026), "MateBook 14s": (2021,2026),
        "MateBook 16s": (2022,2026),
        "Gram 14 2022": (2022,2026), "Gram 15 2022": (2022,2026),
        "Gram 16 2022": (2022,2026), "Gram 17 2022": (2022,2026),
        "Gram 360 14": (2022,2026),
        "Surface Laptop 5 13": (2022,2026), "Surface Laptop 5 15": (2022,2026),
        "Surface Pro 9": (2022,2026), "Surface Laptop Go 2": (2022,2026),
        "Surface Laptop Studio": (2021,2026),
    }
    y_start, y_end = MODEL_YEAR_RANGE.get(model_sel, (2019, 2026))
    all_years      = sorted(f["Year"].unique(), reverse=True)
    valid_years    = [y for y in all_years if y_start <= y <= y_end]
    if not valid_years:
        valid_years = all_years
    year = st.selectbox(L['year'], valid_years)
    condition = st.selectbox(L['condition'], sorted(df_orig["Condition"].unique()))
    screen    = st.selectbox(L['screen'],    sorted(f["Screen_Size"].unique()))

    # GPU — only real options for this brand+model
    gpu_options = sorted(f["GPU"].unique())
    gpu         = st.selectbox(L['gpu'], gpu_options)

    # Touchscreen — disabled with "No" if model never has touchscreen
    touch_vals  = f["Touchscreen"].unique()
    has_touch   = "Yes" in touch_vals
    only_touch  = list(touch_vals) == ["Yes"] or list(touch_vals) == ["Yes"]
    if has_touch:
        touchscreen = st.selectbox(L['touch'], sorted(touch_vals))
    else:
        touchscreen = "No"
        st.selectbox(L['touch'], ["No ❌"], disabled=True, help="This model has no touchscreen")

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button(L['button']):
    input_data = {
        "Brand":        encoders["Brand"].transform([str(brand)])[0],
        "Model":        encoders["Model"].transform([str(model_sel)])[0],
        "RAM":          ram,
        "Storage":      storage,
        "Storage_Type": encoders["Storage_Type"].transform([str(storage_type)])[0],
        "CPU_Gen":      cpu_gen,
        "Year":         year,
        "Condition":    encoders["Condition"].transform([str(condition)])[0],
        "Screen_Size":  screen,
        "GPU":          encoders["GPU"].transform([str(gpu)])[0],
        "Touchscreen":  encoders["Touchscreen"].transform([str(touchscreen)])[0],
    }

    new_data   = pd.DataFrame([input_data])
    mid_price  = model.predict(new_data)[0]
    low_price  = mid_price * 0.88
    high_price = mid_price * 1.12
    fmt = lambda v: f"{v * rate:,.0f}"

    st.markdown(f'<div class="result-title">{L["est_label"]}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="range-grid">
        <div class="range-box low">
            <div class="range-label">{L['low']}</div>
            <div class="range-price">{fmt(low_price)}</div>
            <div class="range-currency">{currency_code}</div>
        </div>
        <div class="range-box mid">
            <div class="range-label">{L['mid']}</div>
            <div class="range-price">{fmt(mid_price)}</div>
            <div class="range-currency">{currency_code}</div>
        </div>
        <div class="range-box high">
            <div class="range-label">{L['high']}</div>
            <div class="range-price">{fmt(high_price)}</div>
            <div class="range-currency">{currency_code}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="footer {rtl}">{L["footer"]}</div>', unsafe_allow_html=True)
