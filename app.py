import streamlit as st
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from supabase import create_client

# =============================
# 🌿 UI STYLES
# =============================
st.markdown("""
<style>
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
# SEITE
# =============================
st.set_page_config(page_title="🌿 Wildpflanzen KI", page_icon="🌱")
st.title("🌿 Wildpflanzen & Bodenanalyse (AI + DB)")
st.write("Lade ein Bild einer Pflanze hoch.")

# =============================
# MODELL
# =============================
@st.cache_resource
def load_model():
    model_name = "marwaALzaabi/plant-identification-vit"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    return processor, model

processor, model = load_model()

# =============================
# 🌿 BOTANISCHES MAPPING (sauber getrennt)
# =============================
def map_plant(label):

    label = label.lower()

    result = {
        "raw": label,
        "db_key": "unbekannt",
        "group": "unbekannt",
        "note": None
    }

    # 🌿 Brennnessel vs Taubnessel
    if "urtica" in label:
        result["db_key"] = "brennnessel"
        result["group"] = "Echte Brennnessel (Urtica)"

    elif "lamium" in label:
        result["db_key"] = "brennnessel"
        result["group"] = "Taubnessel (Lamium)"
        result["note"] = "⚠️ KEINE echte Brennnessel – nur ähnliche Pflanzenfamilie"

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
    res = supabase.table("plants") \
        .select("*") \
        .eq("plant_key", plant_key) \
        .execute()

    return res.data[0] if res.data else None

# =============================
# UPLOAD
# =============================
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    st.write("🔍 Analysiere Pflanze...")

    # =============================
    # KI PREDICTION
    # =============================
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

    st.success(f"🌿 Top-Erkennung: {raw_label} ({round(confidence*100,2)}%)")

    # =============================
    # MAPPING
    # =============================
    mapped = map_plant(raw_label)
    plant_key = mapped["db_key"]
    plant_data = get_plant_data(plant_key) if plant_key != "unbekannt" else None

    st.subheader("🌱 Pflanzen-Einordnung")
    st.write("🔬 Art:", mapped["raw"])
    st.write("🌿 Gruppe:", mapped["group"])

    if mapped["note"]:
        st.warning(mapped["note"])

    st.info(f"DB-Key: {plant_key}")

    # =============================
    # 🧠 UI LOGIK (CLEAN)
    # =============================

    # ❌ UNSICHER
    if plant_key == "unbekannt":

        st.markdown(f"""
        <div class="status-box warning">
        ⚠️ <b>Unsichere Erkennung</b><br><br>
        Die Pflanze konnte nicht eindeutig zugeordnet werden.<br>
        Bitte anderes Bild versuchen.
        </div>
        """, unsafe_allow_html=True)

    # 🌿 ÄHNLICH, ABER KEIN DB MATCH
    elif plant_data is None:

        st.markdown(f"""
        <div class="status-box error">
        🌿 <b>Pflanze erkannt – keine exakte Datenbank-Entsprechung</b><br><br>

        🔬 Erkannt: <b>{mapped['group']}</b><br>
        ⚠️ Hinweis: Diese Pflanze ist nicht direkt in der Datenbank hinterlegt.<br>
        👉 Es wird eine ähnliche Pflanzenkategorie als Referenz genutzt.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🌱 Referenz-Bodenanalyse")
        st.write("⚠️ basiert auf ähnlicher Pflanzenklasse:", plant_key)
        st.write("Kohl, Tomate, Gurke")

    # 🌿 EXAKTER TREFFER
    else:

        st.markdown(f"""
        <div class="status-box success">
        🌿 <b>Pflanze erkannt & zugeordnet</b><br><br>
        {mapped['group']}<br>
        Exakter Datenbankeintrag vorhanden.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("🌱 Bodenanalyse")
        st.write("Boden:", plant_data.get("soil"))
        st.write("Feuchtigkeit:", plant_data.get("moisture"))
        st.write("Sonne:", plant_data.get("sun"))

        st.subheader("🌿 Empfehlungen")
        st.success(plant_data.get("recommendations"))
