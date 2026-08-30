import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Blood Cell Classification",
    page_icon="🩸",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Model loading
#
# IMPORTANT: every saved model already has its own preprocessing (rescaling /
# mobilenet_v2.preprocess_input / resnet50.preprocess_input) baked in as the
# first step of the model graph, right after the Input layer (see the model-
# building cells in the training notebook). That means each model expects
# RAW pixels in [0, 255] as input -- do NOT preprocess externally, or the
# image gets scaled twice and every input collapses to nearly the same
# (wrong) prediction.
# ---------------------------------------------------------------------------

MODEL_FILES = {
    "MobileNetV2 (fine-tuned)": "mobilenet_finetuned_best.keras",
    "ResNet50 (frozen transfer)": "resnet50_best.keras",
    "Custom CNN (baseline)": "baseline_model_best.keras",
}


@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)


# Class names mapping (alphabetical order as per Keras train_ds.class_names)
CLASS_NAMES = [
    'basophil',
    'eosinophil',
    'erythroblast',
    'ig',
    'lymphocyte',
    'monocyte',
    'neutrophil',
    'platelet'
]

st.title("🩸 Blood Cell Image Classification")
st.write("Upload a blood cell microscopic image to predict its class and view confidence scores.")

# Sidebar setup
st.sidebar.header("Model Settings & Info")
model_type = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

try:
    model = load_model(MODEL_FILES[model_type])
except Exception as e:
    st.sidebar.error(
        f"Could not load '{MODEL_FILES[model_type]}'. Make sure this file is "
        f"in the same folder as app.py.\n\n{e}"
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing: resize only. Each model applies its own internal
    # rescaling / preprocess_input as its first layer, so we pass raw
    # [0, 255] pixel values straight in.
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)

    # Run Inference
    predictions = model.predict(img_batch)[0]

    # Defensive check in case a model ever outputs raw logits instead of
    # softmax probabilities (not expected here, since every model ends in
    # a Dense(8, activation='softmax') layer -- kept as a safety net only).
    if np.max(predictions) > 1.0 or np.min(predictions) < 0.0:
        predictions = tf.nn.softmax(predictions).numpy()

    top_pred_idx = np.argmax(predictions)
    top_class = CLASS_NAMES[top_pred_idx]
    top_confidence = predictions[top_pred_idx] * 100

    with col2:
        st.subheader("Classification Result")
        st.success(f"**Predicted Class:** {top_class.capitalize()}")
        st.metric(label="Confidence Score", value=f"{top_confidence:.2f}%")

        # Display probability distribution breakdown
        st.subheader("Class Probabilities")
        prob_dict = {class_name: float(
            prob) for class_name, prob in zip(CLASS_NAMES, predictions)}
        st.bar_chart(prob_dict)
