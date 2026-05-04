import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from supabase import create_client

# =============================
# CONFIG (nur Layout erweitert)
# =============================
st.set_page_config(page_title="🌿 Wildpflanzen KI", page_icon="🌱", layout="wide")

# =============================
# 🌿 ROOTWISE DESIGN (übertragen)
# =============================
st.markdown("""
<style>

/* Hintergrund */
.stApp {
    background-color: #E8F5E9;
    color: black;
}

/* global text schwarz */
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
    color: black;
    text-shadow: 0px 2px 0px rgba(0,0,0,0.15);
}

.hero-subtitle {
    text-align: center;
    font-size: 18px;
    margin-top: 5px;
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

/* SOIL CARD */
.soil-card {
    background-color: #D6EBFF;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: black;
}

/* RECOMMENDATION */
.recommendation-card {
    background-color: #FDECEF;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: black;
}

/* LABELS */
.label {
    font-weight: 700;
    display: block;
    margin-bottom: 4px;
}

.value {
    font-weight: 400;
    opacity: 0.9;
}

/* STATUS BOX (aus deiner ersten App behalten, nur angepasst) */
.status-box {
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
    font-size: 16px;
}

.success {
    background-color: #e6f4ea;
    border-left: 6px solid #2e7d32;
}

.warning {
    background-color: #fff8e1;
    border-left: 6px solid #f9a825;
}

.error {
    background-color: #fdecea;
    border-left: 6px solid #c62828;
}

</style>
""", unsafe_allow_html=True)

# =============================
# SUPABASE
# =============================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================
# HEADER (neu)
# =============================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Wildpflanzen KI</div>
    <div class="hero-subtitle">Wildpflanzen scannen. Boden verstehen.</div>
</div>
""", unsafe_allow_html=True)

st.title("🌿 Wildpflanzen & Bodenanalyse (AI + DB)")
st.write("Lade ein Bild einer Pflanze hoch.")

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
# BOTANICAL MAPPING (unverändert)
# =============================
def map_plant(label):

    label = label.lower()

    result = {
        "raw": label,
        "db_key": "unbekannt",
        "group": "unbekannt",
        "note": None
    }

    if "urtica" in label:
        result["db_key"] = "brennnessel"
        result["group"] = "Echte Brennnessel (Urtica)"

    elif "lamium" in label:
        result["db_key"] = "brennnessel"
        result["group"] = "Taubnessel (Lamium)"
        result["note"] = "⚠️ KEINE echte Brennnessel"

    elif "taraxacum" in label:
        result["db_key"] = "loewenzahn"
        result["group"] = "Löwenzahn"

    elif "trifolium" in label:
        result["db_key"] = "klee"
        result["group"] = "Klee"

    elif "calluna" in label:
        result["db_key"] = "heidekraut"
        result["group"] = "Heidekraut"

    elif "thymus" in label:
        result["db_key"] = "thymian"
        result["group"] = "Thymian"

    elif "matricaria" in label or "chamomilla" in label:
        result["db_key"] = "kamille"
        result["group"] = "Kamille"

    elif "dryopteris" in label or "pteridium" in label:
        result["db_key"] = "farn"
        result["group"] = "Farn"

    elif "achillea" in label:
        result["db_key"] = "schafgabe"
        result["group"] = "Schafgarbe"

    elif "caltha" in label:
        result["db_key"] = "sumpfdotterblume"
        result["group"] = "Sumpfdotterblume"

    elif "carex" in label:
        result["db_key"] = "seggen"
        result["group"] = "Seggen"

    return result

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

    st.write("🔍 Analysiere Pflanze...")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    topk = torch.topk(probs, 3)

    labels = [model.config.id2label[i.item()] for i in topk.indices[0]]
    scores = topk.values[0]

    raw_label = labels[0]
    confidence = float(scores[0])

    st.subheader("🌿 Ergebnisse")

    for label, score in zip(labels, scores):
        st.write(f"👉 {label} ({round(score.item()*100,2)}%)")

    st.success(f"Top-Erkennung: {raw_label} ({round(confidence*100,2)}%)")

    # =============================
    # MAPPING
    # =============================
    mapped = map_plant(raw_label)
    plant_key = mapped["db_key"]
    plant_data = get_plant_data(plant_key) if plant_key != "unbekannt" else None

    st.subheader("🌱 Pflanzen-Einordnung")
    st.write("Art:", mapped["raw"])
    st.write("Gruppe:", mapped["group"])

    if mapped["note"]:
        st.warning(mapped["note"])

    st.info(f"DB-Key: {plant_key}")

    # =============================
    # UI LOGIK (nur Design angepasst)
    # =============================

    if plant_key == "unbekannt":

        st.markdown("""
        <div class="status-box warning">
        ⚠️ <b>Unsichere Erkennung</b><br><br>
        Pflanze konnte nicht eindeutig zugeordnet werden.
        </div>
        """, unsafe_allow_html=True)

    elif plant_data is None:

        st.markdown(f"""
        <div class="status-box error">
        🌿 <b>Pflanze erkannt – kein DB Eintrag</b><br><br>
        {mapped['group']}
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown(f"""
        <div class="status-box success">
        🌿 <b>Pflanze erkannt & zugeordnet</b><br><br>
        {mapped['group']}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🌱 Bodenanalyse")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="soil-card">
                <span class="label">Boden</span>
                <span class="value">{plant_data.get("soil")}</span>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="soil-card">
                <span class="label">Feuchtigkeit</span>
                <span class="value">{plant_data.get("moisture")}</span>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="soil-card">
                <span class="label">Sonne</span>
                <span class="value">{plant_data.get("sun")}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🌿 Empfehlungen")

        st.markdown(f"""
        <div class="recommendation-card">
            {plant_data.get("recommendations")}
        </div>
        """, unsafe_allow_html=True)
