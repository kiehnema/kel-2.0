import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
from supabase import create_client

# =============================
# SUPABASE
# =============================
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =============================
# EINSTELLUNGEN
# =============================
CONFIDENCE_THRESHOLD = 0.7

# =============================
# SEITE
# =============================
st.set_page_config(page_title="🌿 Pflanzen KI", layout="centered")

st.title("🌿 Pflanzen & Bodenanalyse")

# =============================
# MODELL LADEN
# =============================
@st.cache_resource
def load_model_and_labels():
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    with open("labels.txt", "r") as f:
        class_names = f.readlines()
    return model, class_names

model, class_names = load_model_and_labels()

# =============================
# NORMALISIERUNG
# =============================
def normalize(label):
    label = label.lower().strip()

    if "brennnessel" in label or "urtica" in label:
        return "brennnessel"
    if "löwenzahn" in label or "taraxacum" in label:
        return "loewenzahn"
    if "klee" in label or "trifolium" in label:
        return "klee"
    if "lamium" in label:
        return "taubnessel"
    if "schafgarbe" in label:
        return "schafgarbe"
    if "thymian" in label:
        return "thymian"
    if "kamille" in label:
        return "kamille"
    if "distel" in label:
        return "distel"
    if "farn" in label:
        return "farn"
    if "heidekraut" in label:
        return "heidekraut"

    return "unbekannt"

# =============================
# SUPABASE ABFRAGE
# =============================
def get_plant_data(plant_key):
    res = supabase.table("plants") \
        .select("*") \
        .eq("plant_key", plant_key) \
        .execute()

    if res.data:
        return res.data[0]
    return None

# =============================
# UPLOAD
# =============================
uploaded_file = st.file_uploader("📷 Pflanze hochladen", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except:
        st.error("❌ Bild konnte nicht geladen werden.")
        st.stop()

    st.image(image, caption="Dein Bild", use_column_width=True)

    # ----------------------------
    # PREPROCESSING
    # ----------------------------
    size = (224, 224)
    image_model = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image_model)
    normalized = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    # ----------------------------
    # PREDICTION
    # ----------------------------
    with st.spinner("🔍 Analysiere Pflanze..."):
        prediction = model.predict(data)

    index = np.argmax(prediction)
    raw_label = class_names[index].strip()
    confidence = float(prediction[0][index])

    # ----------------------------
    # CONFIDENCE LOGIK
    # ----------------------------
    if confidence >= CONFIDENCE_THRESHOLD:
        st.success(f"🌿 Erkannt: {raw_label}")
        plant_key = normalize(raw_label)
    else:
        st.error("⚠️ Unsichere Erkennung!")
        st.info("👉 Bitte näher aufnehmen oder bessere Beleuchtung nutzen")
        plant_key = "unbekannt"

    st.write(f"Sicherheit: {round(confidence * 100, 2)} %")

    # ----------------------------
    # NUR BEI SICHERER ERKENNUNG
    # ----------------------------
    if confidence >= CONFIDENCE_THRESHOLD:

        st.subheader("🌱 Erkannte Pflanzenklasse")
        st.info(plant_key)

    # ----------------------------
    # SUPABASE DATEN LADEN
    # ----------------------------
    if plant_key != "unbekannt":

        plant_data = get_plant_data(plant_key)

    else:
        st.warning("⚠️ Keine sichere Erkennung → keine Bodenanalyse")
        plant_data = None

    # ----------------------------
    # AUSGABE
    # ----------------------------
    if plant_data:

        st.subheader("🌱 Bodenanalyse (aus Datenbank)")
        st.write("Boden:", plant_data["soil"])
        st.write("Feuchtigkeit:", plant_data["moisture"])
        st.write("Sonne:", plant_data["sun"])

        st.subheader("🌿 Empfehlungen")
        st.success(plant_data["recommendations"])

    else:
        st.warning("❌ Keine Daten in Supabase gefunden für diese Pflanze")
