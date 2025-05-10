import torch
from flask import Flask, request, render_template
from transformers import CLIPProcessor, CLIPModel
from PIL import Image as PILImage
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import ast
import os
import sys


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load product data and embeddings
csv_file_path = 'products_cleaned.csv'
df = pd.read_csv(csv_file_path)
df['categories'] = df['categories'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
df['variations'] = df['variations'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
df['combined_text'] = df['title'].fillna('') + " " + df['description'].fillna('') + " " + df['top_review'].fillna('')
retriever = SentenceTransformer('all-MiniLM-L6-v2')
product_embeddings = retriever.encode(df['combined_text'].tolist(), convert_to_tensor=True)

# Load CLIP model for image analysis
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to analyze uploaded image
def analyze_image(image_path):
    image = PILImage.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return image_features

# Function to match image to products
def match_image_to_products(image_features, top_k=3):
    product_texts = df['combined_text'].tolist()
    text_inputs = clip_processor(text=product_texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**text_inputs)
    similarities = util.pytorch_cos_sim(image_features, text_features)[0]
    top_indices = similarities.argsort(descending=True)[:top_k]
    return df.iloc[top_indices]

# Text-based product retrieval
def retrieve_products(query, top_k=3):
    query_embedding = retriever.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, product_embeddings)[0]
    top_indices = scores.argsort(descending=True)[:top_k]
    return df.iloc[top_indices]

# Generate response HTML
def generate_response(query, relevant_products):
    if relevant_products.empty:
        return "<p>Sorry, I couldn’t find any products matching your request.</p>"
    html_content = f"<h3>Results for '{query}':</h3><ul>"
    for _, product in relevant_products.iterrows():
        price = product['final_price'] if pd.notna(product['final_price']) else "Price not listed"
        currency = product['currency'] if pd.notna(product['currency']) else "USD"
        rating = product['rating'] if pd.notna(product['rating']) else "N/A"
        image_url = product['image_url'] if pd.notna(product['image_url']) else "https://via.placeholder.com/150"
        html_content += (
            f"<li><strong>{product['title']}</strong><br>"
            f"Price: {currency} {price}, Rating: {rating}/5<br>"
            f"<img src='{image_url}' width='150'><br>"
            f"<a href='{product['url']}' target='_blank'>Product Link</a></li><br>"
        )
    html_content += "</ul>"
    return html_content

# Main assistant function
def shopping_assistant(query=None, image_path=None):
    if image_path:
        image_features = analyze_image(image_path)
        relevant_products = match_image_to_products(image_features)
        return generate_response("Matching decor/furniture for your room", relevant_products)
    elif query:
        if "price" in query.lower() and "under" in query.lower():
            max_price = float(query.split("under")[-1].split()[0])
            relevant_products = retrieve_products(query)
            relevant_products = relevant_products[pd.to_numeric(relevant_products['final_price'], errors='coerce') <= max_price]
        else:
            relevant_products = retrieve_products(query)
        return generate_response(query, relevant_products)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/assistant', methods=['POST'])
def assistant():
    query = request.form.get('query')
    image = request.files.get('image')
    if image:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
        image.save(image_path)
        response = shopping_assistant(image_path=image_path)
    elif query:
        response = shopping_assistant(query=query)
    else:
        response = "<p>Please provide a query or upload an image.</p>"
    return response

if __name__ == '__main__':
    app.run(debug=True) 
