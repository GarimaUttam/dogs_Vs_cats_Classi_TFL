import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("model.h5")

st.title("Dog Vs Cat image classification system")
st.write("Upload the image of either cat or dog to predict")

uploaded_imgFile = st.file_uploader("Choose an image of cat or dog", type = ["jpg", "png", "jpeg"])

if uploaded_imgFile is not None:
    # image = Image.open(uploaded_imgFile).convert("L")
    image = Image.open(uploaded_imgFile)
    image = image.resize((224, 224))
    image_arr = np.array(image)/225.0
    image_arr = image_arr.reshape(1, 224, 224, 1)

    st.image(image, caption = "Uploaded image", use_container_width=True)

    prediction = model.predict(image_arr)
    predicted_image = np.argmax(prediction)

    if predicted_image == 0:
        st.write("predicted image is CAT")
    else:
        st.write(f"Predicted image is DOG")