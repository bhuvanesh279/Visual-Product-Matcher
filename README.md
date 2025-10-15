# Product Similarity Search Web App

This is a **Streamlit-based web application** to search for similar products using images.  
The app extracts features from product images using **ResNet50** and compares them via **cosine similarity**.

**Dataset Source:** [Kaggle - Images for Watches, Shoes, Headsets, Laptops](https://www.kaggle.com/datasets/cliffordlee96/images-for-watches-shoes-headsets-laptops)  
Note : Due to GitHub’s upload size limits, only a few sample images are included in this repository.  
The full dataset can be accessed from the Kaggle link above.

## Features
- Upload a product image and find top 5 visually similar products.
- Supports multiple categories (e.g., watches, shoes, laptops, headphones).
- Works directly with a dataset stored in Google Drive.

## Results

**Query Image:**  
![query].

**Top Similar Products:**
1. sneakers 265 (sneakers) - Score: 0.621 ![sneaker1]
2. sneakers 212 (sneakers) - Score: 0.617 ![sneaker2]
3. sneakers 235 (sneakers) - Score: 0.612 ![sneaker3]
"# Visual-Product-Matcher"

## Live Demo
Check out the live app here: https://visual-appuct-matcher-b3nj2ccf32kcnzgjpvmvuy.streamlit.app/



