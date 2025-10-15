import os
import streamlit as st
from PIL import Image

# --- Title ---
st.title("🖼️ Visual Product Matcher")

# --- Define the correct base path ---
base_path = os.path.join(os.path.dirname(__file__), "sample_images")

# --- Check if folder exists ---
if not os.path.exists(base_path):
    st.error("❌ 'sample_images' folder not found. Please ensure it exists in the same directory as app.py.")
else:
    # --- Get all valid image files ---
    image_files = [f for f in os.listdir(base_path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # --- Check if there are any images ---
    if len(image_files) == 0:
        st.warning("⚠️ No sample images found in 'sample_images/' folder.")
    else:
        st.success(f"✅ Found {len(image_files)} sample images.")

        # --- Display images in a grid layout ---
        cols = st.columns(3)
        for i, img_name in enumerate(image_files):
            img_path = os.path.join(base_path, img_name)
            try:
                img = Image.open(img_path)
                with cols[i % 3]:
                    st.image(img, caption=img_name, use_container_width=True)
            except Exception as e:
                st.error(f"Error loading {img_name}: {e}")
