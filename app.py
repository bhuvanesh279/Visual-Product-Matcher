import streamlit as st
import os, numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

st.title("🖼️ Visual Product Matcher")

# === Folder containing sample images ===
base_path = os.path.join(os.path.dirname(__file__), "sample_images")

# === Validate folder ===
if not os.path.exists(base_path):
    st.error("❌ 'sample_images' folder not found in repository.")
    st.stop()

# === Load sample images ===
products = []
for idx, img_file in enumerate(os.listdir(base_path), start=1):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        products.append({
            "id": idx,
            "name": f"Product {idx}",
            "image_path": os.path.join(base_path, img_file)
        })

if len(products) == 0:
    st.warning("⚠️ No sample images found in 'sample_images' folder.")
    st.stop()

st.write(f"✅ Total products loaded: {len(products)}")

# === Load model ===
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return Model(inputs=base_model.input, outputs=base_model.output)

model = load_model()

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB').resize((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten()

# === Precompute all product features ===
@st.cache_data
def compute_dataset_features():
    feats = []
    for p in products:
        feats.append(extract_features(p["image_path"]))
    return np.array(feats)

product_features = compute_dataset_features()

# === Upload and Search ===
uploaded_file = st.file_uploader("📤 Upload a product image", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file)
    st.image(query_img, caption="Query Image", use_container_width=True)

    query_features = extract_features(uploaded_file).reshape(1, -1)
    similarities = cosine_similarity(query_features, product_features)[0]
    sorted_idx = np.argsort(similarities)[::-1]

    st.subheader("🔍 Top Similar Products")
    for idx in sorted_idx[:5]:
        p = products[idx]
        st.image(Image.open(p["image_path"]).resize((200,200)), 
                 caption=f"{p['name']} — Similarity: {similarities[idx]:.3f}")
