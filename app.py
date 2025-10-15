import streamlit as st
from PIL import Image
import os

# Path to sample images
base_path = "sample_images"

st.title("Product Similarity Search")

uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "png"])
if uploaded_file is not None:
    query_image = Image.open(uploaded_file)
    st.image(query_image, caption="Query Image", use_column_width=True)

    # Demo: show all sample images as "similar products"
    st.subheader("Top Similar Products:")
    for img_name in os.listdir(base_path):
        img_path = os.path.join(base_path, img_name)
        img = Image.open(img_path)
        st.image(img, caption=img_name, width=150)

