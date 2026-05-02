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
HIGH_CONFIDENCE = 0.70
MID_CONFIDENCE = 0.50

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
# SUPABASE
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
# UI
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

    plant_key = normalize(raw_label)

    st.write(f"🔎 Modell: {raw_label}")
    st.write(f"📊 Sicherheit: {round(confidence * 100, 2)} %")

    # =====================================================
    # 🔥 LOGIK 1: HOHE SICHERHEIT (≥ 70%)
    # =====================================================
    if confidence >= HIGH_CONFIDENCE and plant_key != "unbekannt":

        st.success(f"🌿 Sicher erkannt: {plant_key}")

        plant_data = get_plant_data(plant_key)

        if plant_data:
            st.subheader("🌱 Bodenanalyse")
            st.write("Boden:", plant_data["soil"])
            st.write("Feuchtigkeit:", plant_data["moisture"])
            st.write("Sonne:", plant_data["sun"])

            st.subheader("🌿 Empfehlungen")
            st.success(plant_data["recommendations"])
        else:
            st.warning("Keine Daten in Supabase gefunden.")

    # =====================================================
    # ⚠️ LOGIK 2: MITTLERE SICHERHEIT (50–70%)
    # =====================================================
    elif confidence >= MID_CONFIDENCE:

        st.warning("⚠️ Unsichere Erkennung – mögliche Pflanzen:")

        # Top 3 Vorschläge anzeigen
        top_indices = np.argsort(prediction[0])[::-1][:3]

        options = []
        mapping = {}

        for i in top_indices:
            label = class_names[i].strip()
            conf = float(prediction[0][i])
            key = normalize(label)

            if key != "unbekannt":
                text = f"{key} ({round(conf*100, 1)}%)"
                options.append(text)
                mapping[text] = key

        if options:
            choice = st.selectbox("Welche Pflanze passt?", options)

            if st.button("🌱 Auswahl bestätigen & Analyse starten"):
                plant_key = mapping[choice]

                plant_data = get_plant_data(plant_key)

                if plant_data:
                    st.subheader("🌱 Bodenanalyse")
                    st.write("Boden:", plant_data["soil"])
                    st.write("Feuchtigkeit:", plant_data["moisture"])
                    st.write("Sonne:", plant_data["sun"])

                    st.subheader("🌿 Empfehlungen")
                    st.success(plant_data["recommendations"])

        else:
            st.error("Keine sinnvollen Vorschläge gefunden.")

    # =====================================================
    # ❌ LOGIK 3: UNTER 50%
    # =====================================================
    else:
        st.error("❌ Zu unsicher erkannt (<50%)")
        st.info("Bitte besseres Bild aufnehmen (Licht, Nähe, Fokus)")
