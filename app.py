import streamlit as st
import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity

st.title("🛍️ Visual Product Matcher")
st.write("Upload an image to find visually similar products from sample images.")

# === Path to sample images inside repo ===
base_path = os.path.join(os.path.dirname(__file__), "sample_images")

# === Load all images with categories ===
products = []
id_counter = 1

for category in os.listdir(base_path):
    category_path = os.path.join(base_path, category)
    if os.path.isdir(category_path):
        for img_file in os.listdir(category_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                products.append({
                    "id": id_counter,
                    "name": os.path.splitext(img_file)[0],
                    "category": category,
                    "image_path": os.path.join(category_path, img_file)
                })
                id_counter += 1

if not products:
    st.error("⚠️ No sample images found inside 'sample_images/' subfolders.")
    st.stop()
else:
    st.success(f"✅ Loaded {len(products)} images across {len(os.listdir(base_path))} categories.")

# === Load ResNet50 Model ===
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten()

# === Upload query image ===
uploaded_file = st.file_uploader("📤 Upload a query image", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file)
    st.image(query_img, caption="Query Image", use_column_width=True)

    query_features = extract_features(uploaded_file).reshape(1, -1)

    # Extract features for all products
    product_features = []
    for p in products:
        product_features.append(extract_features(p['image_path']))
    product_features = np.array(product_features)

    # Compute cosine similarity
    similarities = cosine_similarity(query_features, product_features)[0]
    sorted_idx = np.argsort(similarities)[::-1]

    st.subheader("🔍 Top 5 Similar Products")
    for idx in sorted_idx[:5]:
        p = products[idx]
        st.write(f"**{p['category']}** - {p['name']} (Score: {similarities[idx]:.3f})")
        st.image(Image.open(p['image_path']).resize((200, 200)))
