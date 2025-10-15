# Product Similarity Search Web App

This is a **Streamlit-based web application** to search for similar products using images.  
The app extracts features from product images using **ResNet50** and compares them via **cosine similarity**.

**Dataset Source:** [Kaggle - Images for Watches, Shoes, Headsets, Laptops](https://www.kaggle.com/datasets/cliffordlee96/images-for-watches-shoes-headsets-laptops)  
Note : Due to GitHub’s upload size limits, only a few sample images are included in this repository.  
The full dataset can be accessed from the Kaggle link above.

## Features
-Upload an image of a product.
-Find top visually similar products from the dataset.
-Display similarity scores and images dynamically.
-Lightweight and fast feature extraction using deep learning.

## Technology Stack
-Frontend: Streamlit (Python-based web interface)
-Backend / Feature Extraction: TensorFlow and Keras
-Model: Pretrained ResNet50 (ImageNet weights) for feature embeddings

# Model Details
-Architecture: ResNet50, pretrained on ImageNet
-Input Size: 224 × 224 pixels RGB images
-Feature Extraction: Global average pooling applied to the output of the last convolutional layer
-Output Feature Vector: 2048-dimensional vector for each image
-Similarity Calculation: Cosine similarity between feature vectors
## Results

**Query Image:**  
![query].

**Top Similar Products:**
1. sneakers 265 (sneakers) - Score: 0.621 ![sneaker1]
2. sneakers 212 (sneakers) - Score: 0.617 ![sneaker2]
3. sneakers 235 (sneakers) - Score: 0.612 ![sneaker3]
"# Visual-Product-Matcher"

## Live Demo
Check out the live app here: visual-matcher-bhuvaneshwaran-24mca0085.streamlit.app





