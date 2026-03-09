import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
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
        "chart_title": "📈 Price History by Year",
        "chart_new": "New", "chart_used": "Good Condition",
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
        "chart_title": "📈 تاريخ الأسعار حسب السنة",
        "chart_new": "جديد", "chart_used": "حالة جيدة",
    }
}

FALLBACK_RATES = {
    "🇪🇬 EGP": 1.0, "🇺🇸 USD": 0.01990,
    "🇪🇺 EUR": 0.01710, "🇸🇦 SAR": 0.07450, "🇦🇪 AED": 0.07320,
}

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

# ── Price History Chart ───────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    chart_data = df_orig[
        (df_orig["Brand"] == brand) &
        (df_orig["Model"] == model_sel) &
        (df_orig["RAM"] == ram) &
        (df_orig["Storage"] == storage)
    ].copy()

    if not chart_data.empty:
        # One line per condition with distinct colors
        condition_styles = {
            "New":      {"color": "#00e5ff", "dash": "solid",  "symbol": "circle"},
            "Like New": {"color": "#00ff9d", "dash": "solid",  "symbol": "diamond"},
            "Good":     {"color": "#ffea00", "dash": "dot",    "symbol": "square"},
            "Fair":     {"color": "#ff9f43", "dash": "dashdot","symbol": "triangle-up"},
            "Poor":     {"color": "#ff6b6b", "dash": "dash",   "symbol": "x"},
        }

        fig = go.Figure()

        for cond, style in condition_styles.items():
            cond_data = chart_data[chart_data["Condition"] == cond].groupby("Year")["Price"].mean() * rate
            if not cond_data.empty:
                fig.add_trace(go.Scatter(
                    x=cond_data.index.tolist(),
                    y=cond_data.round(0).tolist(),
                    mode="lines+markers",
                    name=cond,
                    line=dict(color=style["color"], width=2.5, dash=style["dash"]),
                    marker=dict(size=8, color=style["color"], symbol=style["symbol"]),
                ))

        fig.update_layout(
            title_text=f"{L['chart_title']} — {model_sel}",
            title_x=0.5,
            title_font_color="#ffffff",
            title_font_size=14,
            paper_bgcolor="#261e4a",
            plot_bgcolor="#1a1035",
            font_color="#ffffff",
            font_family="Nunito",
            xaxis_title="Year",
            xaxis_gridcolor="#333",
            xaxis_tickmode="linear",
            xaxis_dtick=1,
            xaxis_color="#aaa",
            yaxis_title=f"Price ({currency_code})",
            yaxis_gridcolor="#333",
            yaxis_color="#aaa",
            legend_bgcolor="#261e4a",
            legend_borderwidth=1,
            margin=dict(l=20, r=20, t=50, b=20),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f'<div class="footer {rtl}">{L["footer"]}</div>', unsafe_allow_html=True)
