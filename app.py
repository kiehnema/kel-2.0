import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from supabase import create_client

# =============================
# RAM FIX
# =============================
device = torch.device("cpu")

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Wildpflanzen KI", layout="wide")

# =============================
# DESIGN
# =============================
st.markdown("""
<style>

/* Hintergrund */
.stApp {
    background-color: #E8F5E9;
}

/* 🔥 ALLE SCHRIFTEN SCHWARZ */
* {
    color: #000000 !important;
}

/* Hero */
.hero-container {
    position: sticky;
    top: 0;
    background: linear-gradient(to bottom, #E8F5E9 85%, rgba(232,245,233,0));
    padding-bottom: 10px;
}

.hero-title {
    font-size: 54px;
    font-weight: 800;
    text-align: center;
}

.hero-subtitle {
    text-align: center;
    font-size: 18px;
    opacity: 0.85;
}

/* Button */
.stButton>button {
    background-color: #FADADD;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 16px;
    border: none;
}

/* Upload */
.stFileUploader {
    border: none !important;
    padding: 15px;
    border-radius: 10px;
    background-color: #ffffff;
}

/* Cards */
.soil-card {
    background-color: #D6EBFF;
    padding: 15px;
    border-radius: 12px;
}

.recommendation-card {
    background-color: #FDECEF;
    padding: 15px;
    border-radius: 12px;
}

/* Status Box */
.status-box {
    padding: 15px;
    border-radius: 12px;
    margin-top: 10px;
}

.success {
    background: #e6f4ea;
    border-left: 6px solid #2e7d32;
}

.warning {
    background: #F3F6F4;
    border-left: 6px solid #A8B5A2;
}

.error {
    background: #fdecea;
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
# HEADER
# =============================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Wildpflanzen KI</div>
    <div class="hero-subtitle">Wildpflanzen scannen. Boden verstehen.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h2 style="text-align:left; font-size:28px; margin-top:10px;">
Analyse starten
</h2>
""", unsafe_allow_html=True)

st.markdown("""
<p style="font-size:14px; opacity:0.75; margin-top:5px; margin-bottom:15px;">
Lade ein möglichst klares Bild einer Wildpflanze hoch. Für die beste Erkennung sollten Blätter und Blüte gut sichtbar sein.
</p>
""", unsafe_allow_html=True)

# =============================
# MODEL
# =============================
@st.cache_resource
def load_model():
    model_name = "marwaALzaabi/plant-identification-vit"

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    )

    model.eval()
    model.to(device)

    return processor, model

processor, model = load_model()

# =============================
# MAPPING
# =============================
def map_plant(label):

    label = label.lower()

    if "urtica" in label or "brennnessel" in label:
        return {"db_key": "brennnessel", "group": "Brennnessel"}

    if "taraxacum" in label:
        return {"db_key": "loewenzahn", "group": "Löwenzahn"}

    if "trifolium" in label:
        return {"db_key": "klee", "group": "Klee"}

    if "achillea" in label:
        return {"db_key": "schafgarbe", "group": "Schafgarbe"}

    if "thymus" in label:
        return {"db_key": "thymian", "group": "Thymian"}

    if "matricaria" in label:
        return {"db_key": "kamille", "group": "Kamille"}

    if "cirsium" in label:
        return {"db_key": "distel", "group": "Distel"}

    if "caltha" in label:
        return {"db_key": "sumpfdotterblume", "group": "Sumpfdotterblume"}

    if "carex" in label:
        return {"db_key": "seggen", "group": "Seggen"}

    if "calluna" in label:
        return {"db_key": "heidekraut", "group": "Heidekraut"}

    if "dryopteris" in label:
        return {"db_key": "farn", "group": "Farn"}

    return {"db_key": "unbekannt", "group": "unbekannt"}

# =============================
# DB
# =============================
def get_plant_data(plant_key):
    res = supabase.table("plants").select("*").eq("plant_key", plant_key).execute()
    return res.data[0] if res.data else None

# =============================
# UPLOAD
# =============================
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file)
    image.thumbnail((512, 512))
    image = image.convert("RGB")

    # 📌 ZENTRIERTES BILD
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.image(image, width=350)

    st.write("Analyse läuft...")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    topk = torch.topk(probs, k=3)

    labels = [model.config.id2label[i.item()] for i in topk.indices[0]]
    scores = topk.values[0]

    top3 = list(zip(labels, scores))

    raw_label = top3[0][0]
    confidence = float(top3[0][1])

    st.subheader("Ergebnisse")

    for i, (l, s) in enumerate(top3):
        if i == 0:
            st.markdown(f"**{l} ({round(float(s)*100,2)}%) — Top-Auswahl**")
        else:
            st.write(f"{l} ({round(float(s)*100,2)}%)")

    mapped = map_plant(raw_label)
    plant_key = mapped["db_key"]

    plant_data = None

    if confidence < 0.50:

        st.markdown("""
        <div class="status-box warning">
        Diese Pflanze ist noch nicht in unserer Datenbank vorhanden. Daher können wir aktuell keine Empfehlungen geben.
        </div>
        """, unsafe_allow_html=True)

    elif confidence < 0.70:

        st.warning("Mittlere Sicherheit – Auswahl nötig")

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

    else:

        st.success("Sicher erkannt")

        plant_data = get_plant_data(plant_key)

    if plant_data:

        st.markdown(f"""
        <div class="status-box success">
        Kategorie: <b>{mapped['group']}</b> ({plant_key})
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Bodenanalyse")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"<div class='soil-card'><b>Boden</b><br>{plant_data['soil']}</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='soil-card'><b>Feuchtigkeit</b><br>{plant_data['moisture']}</div>", unsafe_allow_html=True)

        with col3:
            st.markdown(f"<div class='soil-card'><b>Sonne</b><br>{plant_data['sun']}</div>", unsafe_allow_html=True)

        st.markdown("### Empfehlungen")

        st.markdown(f"""
        <div class="recommendation-card">
        {plant_data['recommendations']}
        </div>
        """, unsafe_allow_html=True)

    elif confidence >= 0.50:
        st.markdown("""
        <div class="status-box warning">
        Diese Pflanze ist noch nicht in unserer Datenbank vorhanden. Daher können wir aktuell keine Empfehlungen geben.
        </div>
        """, unsafe_allow_html=True)
