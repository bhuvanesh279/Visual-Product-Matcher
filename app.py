import streamlit as st
import os
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# --------- CONFIG ---------
st.set_page_config(page_title="Product Similarity App", layout="wide")

st.title("Product Similarity Search")

# Use sample images folder
BASE_PATH = "sample_images"

# --------- LOAD MODEL ---------
@st.cache_resource
def load_model():
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    return model

model = load_model()

# --------- HELPER FUNCTIONS ---------
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features

# Precompute features for sample images
@st.cache_data
def load_sample_features(base_path):
    features_dict = {}
    if not os.path.exists(base_path):
        return features_dict
    for img_name in os.listdir(base_path):
        img_path = os.path.join(base_path, img_name)
        try:
            features = extract_features(img_path)
            features_dict[img_name] = features
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    return features_dict

sample_features = load_sample_features(BASE_PATH)

# --------- UPLOAD IMAGE ---------
uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Save uploaded image temporarily
    temp_path = os.path.join("temp_upload.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extract features
    query_features = extract_features(temp_path)
    
    # Compute similarities
    similarities = {}
    for img_name, features in sample_features.items():
        sim = cosine_similarity(query_features, features)[0][0]
        similarities[img_name] = sim
    
    # Show top 5 similar products
    if similarities:
        st.subheader("Top 5 Similar Products")
        top_imgs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
        for img_name, score in top_imgs:
            st.write(f"{img_name} - Similarity: {score:.3f}")
            img_path = os.path.join(BASE_PATH, img_name)
            st.image(img_path, width=200)
    else:
        st.warning("No sample images found. Please add some images in the 'sample_images/' folder.")
else:
    st.info("Upload an image to find similar products.")
