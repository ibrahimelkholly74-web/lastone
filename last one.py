import streamlit as st
import pandas as pd
import requests
import base64
import io
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="centered")

# ── Language ───────────────────────────────────────────────────────────────────
LANG = {
    "en": {
        "title": "Laptop Price Predictor",
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
        "pdf_btn": "📄 Download PDF Report",
        "wa_btn": "📱 Share on WhatsApp",
        "compare_title": "🔄 Compare Two Laptops",
        "compare_btn": "⚡ Compare Now",
        "laptop_a": "Laptop A", "laptop_b": "Laptop B",
        "admin_title": "🔧 Admin Dashboard",
        "admin_pass": "Admin Password",
        "admin_login": "Login",
        "total_preds": "Total Predictions Today",
        "top_brand": "Most Searched Brand",
        "top_model": "Most Searched Model",
        "pred_log": "Prediction Log",
        "admin_wrong": "❌ Wrong password",
        "tabs": ["🏠 Predict", "🔄 Compare", "🔧 Admin"],
    },
    "ar": {
        "title": "توقع سعر اللاب توب",
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
        "pdf_btn": "📄 تحميل تقرير PDF",
        "wa_btn": "📱 مشاركة على واتساب",
        "compare_title": "🔄 مقارنة لاب توبين",
        "compare_btn": "⚡ قارن الآن",
        "laptop_a": "لاب توب A", "laptop_b": "لاب توب B",
        "admin_title": "🔧 لوحة الإدارة",
        "admin_pass": "كلمة المرور",
        "admin_login": "دخول",
        "total_preds": "إجمالي التوقعات اليوم",
        "top_brand": "أكثر ماركة بحثًا",
        "top_model": "أكثر موديل بحثًا",
        "pred_log": "سجل التوقعات",
        "admin_wrong": "❌ كلمة مرور خاطئة",
        "tabs": ["🏠 توقع", "🔄 مقارنة", "🔧 إدارة"],
    }
}

ADMIN_PASSWORD = "admin2024"

FALLBACK_RATES = {
    "🇪🇬 EGP": 1.0, "🇺🇸 USD": 0.01990,
    "🇪🇺 EUR": 0.01710, "🇸🇦 SAR": 0.07450, "🇦🇪 AED": 0.07320,
}

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

# ── Session state ──────────────────────────────────────────────────────────────
if "pred_log" not in st.session_state:
    st.session_state.pred_log = []
if "admin_logged" not in st.session_state:
    st.session_state.admin_logged = False

# ── Fetch live rates ───────────────────────────────────────────────────────────
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

CURRENCIES, is_live = fetch_live_rates()
model, encoders     = load_model()
df_orig             = load_raw()

# ── PDF Generator ──────────────────────────────────────────────────────────────
def generate_pdf(specs, low, mid, high, currency_code, rate):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    elements = []

    title_style = ParagraphStyle("title", parent=styles["Title"],
                                  fontSize=20, textColor=colors.HexColor("#00ff88"),
                                  backColor=colors.black, alignment=TA_CENTER, spaceAfter=10)
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                  fontSize=10, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=20)
    elements.append(Paragraph("💻 Laptop Price Report", title_style))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%d %B %Y — %H:%M')}", sub_style))
    elements.append(Spacer(1, 0.5*cm))

    # Specs table
    spec_data = [["Specification", "Value"]] + [[k, str(v)] for k, v in specs.items()]
    spec_table = Table(spec_data, colWidths=[7*cm, 9*cm])
    spec_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  colors.HexColor("#001a0d")),
        ("TEXTCOLOR",   (0,0), (-1,0),  colors.HexColor("#00ff88")),
        ("FONTNAME",    (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  11),
        ("BACKGROUND",  (0,1), (-1,-1), colors.HexColor("#050505")),
        ("TEXTCOLOR",   (0,1), (-1,-1), colors.HexColor("#cccccc")),
        ("FONTSIZE",    (0,1), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#0a0a0a"), colors.HexColor("#050505")]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#00ff8833")),
        ("PADDING",     (0,0), (-1,-1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    elements.append(spec_table)
    elements.append(Spacer(1, 1*cm))

    # Price boxes
    fmt = lambda v: f"{v * rate:,.0f} {currency_code}"
    price_data = [
        ["💰 Low Estimate", "🎯 Mid Estimate", "💎 High Estimate"],
        [fmt(low),           fmt(mid),           fmt(high)],
    ]
    price_table = Table(price_data, colWidths=[5.33*cm, 5.33*cm, 5.34*cm])
    price_table.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (0,0), colors.HexColor("#001133")),
        ("BACKGROUND",  (1,0), (1,0), colors.HexColor("#001a0d")),
        ("BACKGROUND",  (2,0), (2,0), colors.HexColor("#001a1a")),
        ("TEXTCOLOR",   (0,0), (0,0), colors.HexColor("#0088ff")),
        ("TEXTCOLOR",   (1,0), (1,0), colors.HexColor("#00ff88")),
        ("TEXTCOLOR",   (2,0), (2,0), colors.HexColor("#00ffcc")),
        ("BACKGROUND",  (0,1), (-1,1), colors.HexColor("#050505")),
        ("TEXTCOLOR",   (0,1), (-1,1), colors.white),
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0),  10),
        ("FONTSIZE",    (0,1), (-1,1),  13),
        ("ALIGN",       (0,0), (-1,-1), "CENTER"),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#00ff8833")),
        ("PADDING",     (0,0), (-1,-1), 12),
    ]))
    elements.append(price_table)
    elements.append(Spacer(1, 1*cm))

    footer_style = ParagraphStyle("footer", parent=styles["Normal"],
                                   fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    elements.append(Paragraph("Powered by Laptop Price Predictor AI · Egyptian Market", footer_style))

    doc.build(elements)
    buf.seek(0)
    return buf.read()

# ── Predict helper ─────────────────────────────────────────────────────────────
def predict_price(brand, model_sel, ram, storage, storage_type,
                  cpu_gen, year, condition, screen, gpu, touchscreen):
    input_data = {
        "Brand":        encoders["Brand"].transform([str(brand)])[0],
        "Model":        encoders["Model"].transform([str(model_sel)])[0],
        "RAM":          ram, "Storage": storage,
        "Storage_Type": encoders["Storage_Type"].transform([str(storage_type)])[0],
        "CPU_Gen":      cpu_gen, "Year": year,
        "Condition":    encoders["Condition"].transform([str(condition)])[0],
        "Screen_Size":  screen,
        "GPU":          encoders["GPU"].transform([str(gpu)])[0],
        "Touchscreen":  encoders["Touchscreen"].transform([str(touchscreen)])[0],
    }
    mid   = model.predict(pd.DataFrame([input_data]))[0]
    return mid * 0.88, mid, mid * 1.12

# ── Laptop form widget ─────────────────────────────────────────────────────────
def laptop_form(key, L):
    brand       = st.selectbox(L['brand'],  sorted(df_orig["Brand"].unique()),  key=f"brand_{key}")
    models_list = sorted(df_orig[df_orig["Brand"] == brand]["Model"].unique())
    model_sel   = st.selectbox(L['model'],  models_list, key=f"model_{key}")
    filtered    = df_orig[(df_orig["Brand"] == brand) & (df_orig["Model"] == model_sel)]
    f           = filtered if not filtered.empty else df_orig
    ram          = st.selectbox(L['ram'],          sorted(f["RAM"].unique()),           key=f"ram_{key}")
    storage      = st.selectbox(L['storage'],      sorted(f["Storage"].unique()),       key=f"storage_{key}")
    storage_type = st.selectbox(L['storage_type'], sorted(f["Storage_Type"].unique()),  key=f"stype_{key}")
    cpu_gen      = st.selectbox(L['cpu'],          sorted(f["CPU_Gen"].unique()),        key=f"cpu_{key}")
    y_start, y_end = MODEL_YEAR_RANGE.get(model_sel, (2019, 2026))
    valid_years    = [y for y in sorted(f["Year"].unique(), reverse=True) if y_start <= y <= y_end] or sorted(f["Year"].unique(), reverse=True)
    year         = st.selectbox(L['year'],      valid_years,                            key=f"year_{key}")
    condition    = st.selectbox(L['condition'], sorted(df_orig["Condition"].unique()),  key=f"cond_{key}")
    screen       = st.selectbox(L['screen'],    sorted(f["Screen_Size"].unique()),      key=f"screen_{key}")
    gpu          = st.selectbox(L['gpu'],       sorted(f["GPU"].unique()),              key=f"gpu_{key}")
    touch_vals   = f["Touchscreen"].unique()
    if "Yes" in touch_vals:
        touchscreen = st.selectbox(L['touch'], sorted(touch_vals), key=f"touch_{key}")
    else:
        touchscreen = "No"
        st.selectbox(L['touch'], ["No ❌"], disabled=True, key=f"touch_{key}")
    return brand, model_sel, ram, storage, storage_type, cpu_gen, year, condition, screen, gpu, touchscreen

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Cairo:wght@400;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, [class*="css"], .stApp {
    font-family: 'Nunito', sans-serif;
    background-color: #000000 !important;
    color: #00ff88 !important;
}
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 860px !important; }

.hero { text-align: center; padding: 2rem 1rem 1rem; }
.hero-icon { font-size: 3rem; display: block; margin-bottom: 0.3rem; animation: bounce 2s infinite; }
@keyframes bounce { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-8px)} }
.hero h1 {
    font-size: 2.2rem; font-weight: 900;
    background: linear-gradient(135deg, #00ff88, #00e5ff, #00ff88);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.15;
}

.live-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #001a0d; border: 1.5px solid #00ff88;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.75rem; font-weight: 700; color: #00ff88;
}
.live-dot { width: 7px; height: 7px; border-radius: 50%; background: #00ff88; animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.2} }

.card {
    background: #0a0a0a; border: 1.5px solid #00ff8844;
    border-radius: 20px; padding: 1.8rem; margin-bottom: 1.2rem;
    box-shadow: 0 0 24px #00ff8811;
}
.card-title {
    font-size: 0.76rem; font-weight: 800; letter-spacing: 0.12em;
    text-transform: uppercase; color: #00ff88; margin-bottom: 1rem;
}

label, .stSelectbox label {
    font-size: 0.80rem !important; font-weight: 700 !important;
    color: #00cc66 !important; letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

div[data-baseweb="select"] > div {
    background-color: #0d0d0d !important; border: 1.5px solid #00ff8844 !important;
    border-radius: 10px !important; color: #00ff88 !important;
    font-size: 0.93rem !important; transition: all 0.2s;
}
div[data-baseweb="select"] > div:hover { border-color: #00ff88 !important; box-shadow: 0 0 10px #00ff8833 !important; }
div[data-baseweb="select"] > div:focus-within { border-color: #00ff88 !important; box-shadow: 0 0 0 3px #00ff8822 !important; }
ul[role="listbox"] { background-color: #0a0a0a !important; border: 1.5px solid #00ff8844 !important; border-radius: 10px !important; }
li[role="option"] { color: #00ff88 !important; }
li[role="option"]:hover { background-color: #001a0d !important; }

.stRadio > div { flex-direction: row !important; gap: 0.5rem; }
.stRadio label { color: #00ff88 !important; font-weight: 700 !important; font-size: 0.88rem !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #0a0a0a !important; border-radius: 14px; padding: 4px; gap: 4px; border: 1.5px solid #00ff8822; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #00ff8888 !important; border-radius: 10px !important; font-weight: 700 !important; font-size: 0.88rem !important; }
.stTabs [aria-selected="true"] { background: #001a0d !important; color: #00ff88 !important; border: 1px solid #00ff8866 !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem !important; }

div.stButton > button {
    width: 100%; padding: 0.85rem 2rem;
    background: #000000 !important;
    color: #00ff88 !important; font-size: 1rem; font-weight: 900; letter-spacing: 0.06em;
    border: 2px solid #00ff88 !important; border-radius: 14px; cursor: pointer;
    transition: all 0.25s ease; box-shadow: 0 0 20px #00ff8822; margin-top: 0.5rem;
}
div.stButton > button:hover {
    background: #00ff88 !important; color: #000000 !important;
    transform: translateY(-2px); box-shadow: 0 0 40px #00ff8866;
}

.stDownloadButton > button {
    width: 100%; padding: 0.8rem 2rem;
    background: #001a0d !important; color: #00ff88 !important;
    border: 2px solid #00ff8866 !important; border-radius: 14px;
    font-size: 0.95rem; font-weight: 800; cursor: pointer;
    transition: all 0.25s ease; box-shadow: 0 0 16px #00ff8811;
}
.stDownloadButton > button:hover { background: #002a14 !important; border-color: #00ff88 !important; }

.stTextInput > div > div > input {
    background: #0d0d0d !important; border: 1.5px solid #00ff8844 !important;
    border-radius: 10px !important; color: #00ff88 !important; font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus { border-color: #00ff88 !important; box-shadow: 0 0 0 3px #00ff8822 !important; }

.stAlert { background: #001a0d !important; border: 1px solid #00ff8844 !important; color: #00ff88 !important; border-radius: 10px !important; }
.stMetric { background: #0a0a0a; border: 1.5px solid #00ff8833; border-radius: 14px; padding: 1rem; }
[data-testid="stMetricValue"] { color: #00ff88 !important; font-size: 2rem !important; font-weight: 900 !important; }
[data-testid="stMetricLabel"] { color: #00cc66 !important; font-size: 0.8rem !important; font-weight: 700 !important; }

.result-title {
    font-size: 0.8rem; font-weight: 800; letter-spacing: 0.12em;
    text-transform: uppercase; color: #00ff88;
    margin: 1.2rem 0 0.8rem; text-align: center;
}
.range-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
.range-box { border-radius: 18px; padding: 1.8rem 1rem; text-align: center; animation: fadeUp 0.5s ease; background: #050505; }
.range-box.low  { border: 2px solid #0088ff; box-shadow: 0 0 24px #0088ff22; }
.range-box.mid  { border: 2px solid #00ff88; box-shadow: 0 0 36px #00ff8844; }
.range-box.high { border: 2px solid #00ffcc; box-shadow: 0 0 24px #00ffcc22; }
.range-label { font-size: 0.7rem; font-weight: 800; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 0.6rem; }
.low  .range-label { color: #0088ff; }
.mid  .range-label { color: #00ff88; }
.high .range-label { color: #00ffcc; }
.range-price { font-size: 1.9rem; font-weight: 900; line-height: 1; }
.low  .range-price { color: #0088ff; }
.mid  .range-price { color: #00ff88; }
.high .range-price { color: #00ffcc; }
.range-currency { font-size: 0.72rem; margin-top: 0.3rem; font-weight: 700; color: #1a5a3a; }

.wa-btn {
    display: block; text-align: center; background: #001a0d;
    border: 2px solid #25d366; border-radius: 14px;
    padding: 0.85rem; color: #25d366 !important; font-weight: 900;
    font-size: 1rem; text-decoration: none; margin-top: 0.8rem;
    transition: all 0.25s; box-shadow: 0 0 16px #25d36622;
}
.wa-btn:hover { background: #25d366; color: #000 !important; box-shadow: 0 0 30px #25d36666; }

.compare-box { background: #050505; border-radius: 18px; padding: 1.5rem; margin-top: 1rem; }
.compare-header { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 0.5rem; margin-bottom: 0.8rem; }
.compare-row { display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 0.5rem; padding: 0.5rem 0; border-bottom: 1px solid #00ff8811; }
.compare-label { color: #00cc66; font-size: 0.8rem; font-weight: 700; }
.compare-val { color: #ffffff; font-size: 0.9rem; font-weight: 700; text-align: center; }
.compare-winner { color: #00ff88; font-weight: 900; font-size: 1.1rem; text-align: center; }

.admin-stat { background: #0a0a0a; border: 1.5px solid #00ff8833; border-radius: 14px; padding: 1.2rem; text-align: center; }
.admin-stat-val { font-size: 2.2rem; font-weight: 900; color: #00ff88; }
.admin-stat-lbl { font-size: 0.75rem; font-weight: 700; color: #00cc66; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 0.2rem; }

@keyframes fadeUp { from{opacity:0;transform:translateY(14px)} to{opacity:1;transform:translateY(0)} }

.rtl { direction: rtl; text-align: right; font-family: 'Cairo', sans-serif !important; }
.rtl label, .rtl .card-title, .rtl .result-title { font-family: 'Cairo', sans-serif !important; letter-spacing: 0 !important; }
.footer { text-align: center; color: #1a4a2a; font-size: 0.78rem; margin-top: 3rem; }
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

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
</div>
""", unsafe_allow_html=True)

if is_live:
    st.markdown(f'<div style="text-align:center;margin-bottom:1rem"><span class="live-badge"><span class="live-dot"></span>{L["live_rates"]}</span></div>', unsafe_allow_html=True)
else:
    st.warning(L["fallback_rates"])

# ── Currency ───────────────────────────────────────────────────────────────────
st.markdown(f'<div class="card {rtl}"><div class="card-title">{L["currency_label"]}</div>', unsafe_allow_html=True)
currency      = st.selectbox("", list(CURRENCIES.keys()), label_visibility="collapsed")
rate          = CURRENCIES[currency]
currency_code = currency.split(" ")[1]
st.markdown('</div>', unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(L["tabs"])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="card {rtl}"><div class="card-title">{L["section"]}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        brand       = st.selectbox(L['brand'], sorted(df_orig["Brand"].unique()), key="p_brand")
        models_list = sorted(df_orig[df_orig["Brand"] == brand]["Model"].unique())
        model_sel   = st.selectbox(L['model'], models_list, key="p_model")
        filtered    = df_orig[(df_orig["Brand"] == brand) & (df_orig["Model"] == model_sel)]
        f           = filtered if not filtered.empty else df_orig
        ram          = st.selectbox(L['ram'],          sorted(f["RAM"].unique()),          key="p_ram")
        storage      = st.selectbox(L['storage'],      sorted(f["Storage"].unique()),      key="p_storage")
        storage_type = st.selectbox(L['storage_type'], sorted(f["Storage_Type"].unique()), key="p_stype")
    with col2:
        cpu_gen     = st.selectbox(L['cpu'],  sorted(f["CPU_Gen"].unique()), key="p_cpu")
        y_start, y_end = MODEL_YEAR_RANGE.get(model_sel, (2019, 2026))
        valid_years    = [y for y in sorted(f["Year"].unique(), reverse=True) if y_start <= y <= y_end] or sorted(f["Year"].unique(), reverse=True)
        year        = st.selectbox(L['year'],      valid_years,                           key="p_year")
        condition   = st.selectbox(L['condition'], sorted(df_orig["Condition"].unique()), key="p_cond")
        screen      = st.selectbox(L['screen'],    sorted(f["Screen_Size"].unique()),     key="p_screen")
        gpu         = st.selectbox(L['gpu'],       sorted(f["GPU"].unique()),              key="p_gpu")
        touch_vals  = f["Touchscreen"].unique()
        if "Yes" in touch_vals:
            touchscreen = st.selectbox(L['touch'], sorted(touch_vals), key="p_touch")
        else:
            touchscreen = "No"
            st.selectbox(L['touch'], ["No ❌"], disabled=True, key="p_touch")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button(L['button'], key="predict_btn"):
        low_price, mid_price, high_price = predict_price(
            brand, model_sel, ram, storage, storage_type,
            cpu_gen, year, condition, screen, gpu, touchscreen
        )
        fmt = lambda v: f"{v * rate:,.0f}"

        # Log prediction
        st.session_state.pred_log.append({
            "time": datetime.now().strftime("%H:%M"),
            "brand": brand, "model": model_sel,
            "condition": condition,
            "price": f"{fmt(mid_price)} {currency_code}"
        })

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

        st.markdown("<br>", unsafe_allow_html=True)

        # ── PDF Download ──────────────────────────────────────────
        specs = {
            "Brand": brand, "Model": model_sel,
            "RAM": f"{ram} GB", "Storage": f"{storage} GB",
            "Storage Type": storage_type, "CPU Gen": cpu_gen,
            "Year": year, "Condition": condition,
            "Screen": f"{screen}\"", "GPU": gpu,
            "Touchscreen": touchscreen,
        }
        pdf_bytes = generate_pdf(specs, low_price, mid_price, high_price, currency_code, rate)
        col_pdf, col_wa = st.columns(2)
        with col_pdf:
            st.download_button(
                label=L["pdf_btn"],
                data=pdf_bytes,
                file_name=f"{brand}_{model_sel}_price.pdf".replace(" ", "_"),
                mime="application/pdf",
                key="dl_pdf"
            )

        # ── WhatsApp Share ────────────────────────────────────────
        with col_wa:
            wa_text = (
                f"💻 *Laptop Price Estimate*\n"
                f"🏷️ {brand} {model_sel}\n"
                f"💾 {ram}GB RAM | {storage}GB {storage_type}\n"
                f"📅 Year: {year} | Condition: {condition}\n"
                f"💰 Low: {fmt(low_price)} {currency_code}\n"
                f"🎯 Mid: {fmt(mid_price)} {currency_code}\n"
                f"💎 High: {fmt(high_price)} {currency_code}\n"
                f"_Powered by Laptop Price Predictor AI_"
            )
            wa_url = f"https://wa.me/?text={requests.utils.quote(wa_text)}"
            st.markdown(f'<a href="{wa_url}" target="_blank" class="wa-btn">📱 {L["wa_btn"]}</a>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="result-title">{L["compare_title"]}</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f'<div class="card {rtl}"><div class="card-title">{L["laptop_a"]}</div>', unsafe_allow_html=True)
        a = laptop_form("a", L)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown(f'<div class="card {rtl}"><div class="card-title">{L["laptop_b"]}</div>', unsafe_allow_html=True)
        b = laptop_form("b", L)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button(L["compare_btn"], key="cmp_btn"):
        la, ma, ha = predict_price(*a)
        lb, mb, hb = predict_price(*b)
        fmt = lambda v: f"{v * rate:,.0f} {currency_code}"

        winner = L["laptop_a"] if ma < mb else (L["laptop_b"] if mb < ma else "—")
        cheaper_by = abs(ma - mb) * rate

        st.markdown(f"""
        <div class="compare-box">
            <div class="compare-header">
                <div class="compare-label"></div>
                <div class="compare-label" style="text-align:center">{L["laptop_a"]}</div>
                <div class="compare-label" style="text-align:center">{L["laptop_b"]}</div>
            </div>
            <div class="compare-row">
                <div class="compare-label">Brand</div>
                <div class="compare-val">{a[0]}</div><div class="compare-val">{b[0]}</div>
            </div>
            <div class="compare-row">
                <div class="compare-label">Model</div>
                <div class="compare-val">{a[1]}</div><div class="compare-val">{b[1]}</div>
            </div>
            <div class="compare-row">
                <div class="compare-label">RAM</div>
                <div class="compare-val">{a[2]} GB</div><div class="compare-val">{b[2]} GB</div>
            </div>
            <div class="compare-row">
                <div class="compare-label">Storage</div>
                <div class="compare-val">{a[3]} GB</div><div class="compare-val">{b[3]} GB</div>
            </div>
            <div class="compare-row">
                <div class="compare-label">Condition</div>
                <div class="compare-val">{a[7]}</div><div class="compare-val">{b[7]}</div>
            </div>
            <div class="compare-row">
                <div class="compare-label">Low</div>
                <div class="compare-val" style="color:#0088ff">{fmt(la)}</div>
                <div class="compare-val" style="color:#0088ff">{fmt(lb)}</div>
            </div>
            <div class="compare-row">
                <div class="compare-label">Mid Price</div>
                <div class="compare-val" style="color:#00ff88;font-size:1.1rem">{fmt(ma)}</div>
                <div class="compare-val" style="color:#00ff88;font-size:1.1rem">{fmt(mb)}</div>
            </div>
            <div class="compare-row">
                <div class="compare-label">High</div>
                <div class="compare-val" style="color:#00ffcc">{fmt(ha)}</div>
                <div class="compare-val" style="color:#00ffcc">{fmt(hb)}</div>
            </div>
            <div style="text-align:center;margin-top:1.2rem;padding-top:1rem;border-top:1px solid #00ff8833">
                <div style="color:#00cc66;font-size:0.75rem;font-weight:800;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem">💰 Better Value</div>
                <div style="color:#00ff88;font-size:1.4rem;font-weight:900">{winner}</div>
                <div style="color:#aaa;font-size:0.82rem;margin-top:0.3rem">cheaper by {fmt(cheaper_by)}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — ADMIN
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="result-title">{L["admin_title"]}</div>', unsafe_allow_html=True)

    if not st.session_state.admin_logged:
        pwd = st.text_input(L["admin_pass"], type="password", key="admin_pwd")
        if st.button(L["admin_login"], key="admin_btn"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_logged = True
                st.rerun()
            else:
                st.error(L["admin_wrong"])
    else:
        logs = st.session_state.pred_log
        total = len(logs)

        # Stats
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="admin-stat"><div class="admin-stat-val">{total}</div><div class="admin-stat-lbl">{L["total_preds"]}</div></div>', unsafe_allow_html=True)
        with c2:
            top_brand = pd.Series([l["brand"] for l in logs]).value_counts().idxmax() if logs else "—"
            st.markdown(f'<div class="admin-stat"><div class="admin-stat-val">{top_brand}</div><div class="admin-stat-lbl">{L["top_brand"]}</div></div>', unsafe_allow_html=True)
        with c3:
            top_model = pd.Series([l["model"] for l in logs]).value_counts().idxmax() if logs else "—"
            st.markdown(f'<div class="admin-stat"><div class="admin-stat-val" style="font-size:1.1rem">{top_model}</div><div class="admin-stat-lbl">{L["top_model"]}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Log table
        if logs:
            st.markdown(f'<div class="card-title" style="margin-bottom:0.8rem">{L["pred_log"]}</div>', unsafe_allow_html=True)
            df_log = pd.DataFrame(logs[::-1])
            st.dataframe(df_log, use_container_width=True, hide_index=True)
        else:
            st.info("No predictions yet today.")

        if st.button("🚪 Logout", key="admin_logout"):
            st.session_state.admin_logged = False
            st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="footer {rtl}">{L["footer"]}</div>', unsafe_allow_html=True)
