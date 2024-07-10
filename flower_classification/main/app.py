import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Function to load and preprocess the image
def load_and_preprocess_image(image_bytes, target_size=(256,256)):
    image = Image.open(image_bytes).convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array


def load_model():
    model = tf.keras.models.load_model('flower.h5')  # Load your trained model
    return model

# Function to make predictions
def predict(image, model):
    predictions = model.predict(image)
    return predictions

def main():
    st.title('Flower Image Classifier')

    # Upload image through Streamlit widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        image = load_and_preprocess_image(uploaded_file)
        st.image(image[0], caption='Uploaded Image', use_column_width=True)

        # Load model
        model = load_model()

        # Make prediction on image
        if st.button('Classify'):
            predictions = predict(image, model)
            class_names=['bougainvillea','daisies','garden_roses','gardenias','hibiscus','hydrangeas','lilies','orchids','peonies','tulip']
            predictions=class_names[np.argmax(predictions)]
            st.write("prediction:")
            st.subheader(predictions)
            

if __name__ == '__main__':
    main()

