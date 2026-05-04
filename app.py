import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from supabase import create_client

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="🌿 Wildpflanzen KI", page_icon="🌱", layout="wide")

# =============================
# 🌿 ROOTWISE DESIGN
# =============================
st.markdown("""
<style>

.stApp {
    background-color: #E8F5E9;
    color: black;
}

html, body, [class*="css"] {
    color: black !important;
}

h1, h2, h3, h4, h5, h6, p, span, div {
    color: black !important;
}

/* HEADER */
.hero-container {
    position: sticky;
    top: 0;
    background: linear-gradient(to bottom, #E8F5E9 85%, rgba(232,245,233,0));
    padding-bottom: 10px;
    z-index: 10;
}

.hero-title {
    font-size: 54px;
    font-weight: 800;
    text-align: center;
    margin-top: 20px;
    margin-bottom: 0px;
    text-shadow: 0px 2px 0px rgba(0,0,0,0.15);
}

.hero-subtitle {
    text-align: center;
    font-size: 18px;
    opacity: 0.85;
}

/* BUTTONS */
.stButton>button {
    background-color: #FADADD;
    color: black !important;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 16px;
    border: none;
}

/* FILE UPLOAD */
.stFileUploader {
    border: 2px dashed #90CAF9;
    padding: 15px;
    border-radius: 10px;
}

/* CARDS */
.soil-card {
    background-color: #D6EBFF;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
}

.recommendation-card {
    background-color: #FDECEF;
    padding: 15px;
    border-radius: 12px;
}

/* STATUS */
.status-box {
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}

.success { background: #e6f4ea; border-left: 6px solid #2e7d32; }
.warning { background: #fff8e1; border-left: 6px solid #f9a825; }
.error { background: #fdecea; border-left: 6px solid #c62828; }

.label { font-weight: 700; }
.value { opacity: 0.9; }

</style>
""", unsafe_allow_html=True)

# =============================
# SUPABASE
# =============================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================
# HEADER
# =============================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Wildpflanzen KI</div>
    <div class="hero-subtitle">Wildpflanzen scannen. Boden verstehen.</div>
</div>
""", unsafe_allow_html=True)

st.title("🌿 Wildpflanzen & Bodenanalyse")
st.write("Bild hochladen und Pflanze analysieren")

# =============================
# MODEL
# =============================
@st.cache_resource
def load_model():
    model_name = "marwaALzaabi/plant-identification-vit"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

# =============================
# MAPPING (FINAL DB READY)
# =============================
def map_plant(label):

    label = label.lower()

    if "urtica" in label or "brennnessel" in label:
        return {"db_key": "brennnessel", "group": "Brennnessel"}

    if "taraxacum" in label or "löwenzahn" in label:
        return {"db_key": "loewenzahn", "group": "Löwenzahn"}

    if "trifolium" in label or "klee" in label:
        return {"db_key": "klee", "group": "Klee"}

    if "achillea" in label:
        return {"db_key": "schafgarbe", "group": "Schafgarbe"}

    if "thymus" in label:
        return {"db_key": "thymian", "group": "Thymian"}

    if "matricaria" in label or "kamille" in label:
        return {"db_key": "kamille", "group": "Kamille"}

    if "distel" in label or "cirsium" in label:
        return {"db_key": "distel", "group": "Distel"}

    if "caltha" in label:
        return {"db_key": "sumpfdotterblume", "group": "Sumpfdotterblume"}

    if "carex" in label or "segge" in label:
        return {"db_key": "seggen", "group": "Seggen"}

    if "calluna" in label:
        return {"db_key": "heidekraut", "group": "Heidekraut"}

    if "dryopteris" in label or "farn" in label:
        return {"db_key": "farn", "group": "Farn"}

    return {"db_key": "unbekannt", "group": "unbekannt"}

# =============================
# SUPABASE
# =============================
def get_plant_data(plant_key):
    res = supabase.table("plants").select("*").eq("plant_key", plant_key).execute()
    return res.data[0] if res.data else None

# =============================
# UPLOAD
# =============================
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    st.write("🔍 Analyse läuft...")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    topk = torch.topk(probs, 3)

    labels = [model.config.id2label[i.item()] for i in topk.indices[0]]
    scores = topk.values[0]

    top3 = list(zip(labels, scores))

    raw_label = top3[0][0]
    confidence = float(top3[0][1])

    st.subheader("🌿 Ergebnisse")

    for l, s in top3:
        st.write(f"👉 {l} ({round(float(s)*100,2)}%)")

    st.success(f"Top: {raw_label} ({round(confidence*100,2)}%)")

    mapped = map_plant(raw_label)
    plant_key = mapped["db_key"]

    plant_data = None

    # =============================
    # < 50% UNSICHER
    # =============================
    if confidence < 0.50:

        st.markdown("""
        <div class="status-box warning">
        ⚠️ Unsichere Erkennung – bitte neues Bild aufnehmen
        </div>
        """, unsafe_allow_html=True)

    # =============================
    # 50–70% AUSWAHL
    # =============================
    elif confidence < 0.70:

        st.warning("⚠️ Mittlere Sicherheit – bitte auswählen")

        options = {}
        choices = []

        for l, s in top3:
            m = map_plant(l)
            if m["db_key"] != "unbekannt":
                text = f"{m['group']} ({round(float(s)*100,1)}%)"
                choices.append(text)
                options[text] = m["db_key"]

        if choices:

            choice = st.selectbox("Auswahl", choices)

            if st.button("Weiter"):

                plant_key = options[choice]
                plant_data = get_plant_data(plant_key)

    # =============================
    # > 70% DIREKT
    # =============================
    else:

        st.success("✔️ Sicher erkannt")

        plant_data = get_plant_data(plant_key)

    # =============================
    # OUTPUT DB
    # =============================
    if plant_data:

        st.markdown("### 🌱 Bodenanalyse")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="soil-card">
            <div class="label">Boden</div>
            <div class="value">{plant_data['soil']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="soil-card">
            <div class="label">Feuchtigkeit</div>
            <div class="value">{plant_data['moisture']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="soil-card">
            <div class="label">Sonne</div>
            <div class="value">{plant_data['sun']}</div>
            </div>
            """, unsafe_allow_html=True)


        st.markdown("### 🌿 Empfehlungen")

        st.markdown(f"""
        <div class="recommendation-card">
        {plant_data['recommendations']}
        </div>
        """, unsafe_allow_html=True)

    elif confidence >= 0.50:
        st.error("Keine Datenbankdaten gefunden")
