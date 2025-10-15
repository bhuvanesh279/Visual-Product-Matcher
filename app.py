# Write the Streamlit code to app.py
app_code = """
import streamlit as st
import pickle, os, numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

st.title("Product Similarity Search")

base_path = os.path.join(os.path.dirname(__file__), "sample_images")  # dataset path
features_path = "features.pkl"  # pre-extracted features

# Load products
products = []
id_counter = 1
for category in os.listdir(base_path):
    cat_path = os.path.join(base_path, category)
    if os.path.isdir(cat_path):
        for img_file in os.listdir(cat_path):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                products.append({
                    "id": id_counter,
                    "name": f"{category} {id_counter}",
                    "category": category,
                    "image_path": os.path.join(cat_path, img_file)
                })
                id_counter += 1

st.write(f"Total products loaded: {len(products)}")

# Load features
product_features = pickle.load(open(features_path, "rb"))

# Load ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB').resize((224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x, verbose=0)
    return features.flatten()

uploaded_file = st.file_uploader("Upload a query image", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    query_img = Image.open(uploaded_file)
    st.image(query_img, caption="Query Image", use_column_width=True)

    query_features = extract_features(uploaded_file).reshape(1, -1)
    similarities = cosine_similarity(query_features, product_features)[0]
    sorted_idx = np.argsort(similarities)[::-1]

    st.subheader("Top Similar Products")
    for idx in sorted_idx[:5]:
        p = products[idx]
        st.write(f"{p['name']} ({p['category']}) - Score: {similarities[idx]:.3f}")
        st.image(Image.open(p['image_path']).resize((200,200)))
"""

with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py created successfully!")
