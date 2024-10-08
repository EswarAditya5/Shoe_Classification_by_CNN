import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the pre-trained model
model = load_model("shoe_classification_model.h5")

# page setup
st.set_page_config(
    page_title="Shoe Classifier", 
    page_icon="👟", 
)

# Define the shoe categories
categories = ['Boots', 'Sandals', 'Slippers']  # Update with your categories

# st.title('Shoe Category Classifier')
st.title('Shoe Type Identifier')

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")

    # Preprocess the image
    img_array = np.array(image)
    img_array = tf.image.resize(img_array, [224, 224])
    img_array = tf.expand_dims(img_array, 0) / 255.0  # Normalization

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]

    st.write(f"Prediction: {predicted_class}")
