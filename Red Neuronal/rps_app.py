import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np
model = tf.keras.models.load_model('save_at_68.h5')


st.write("""
         # Red Neuronal de Reconocimiento de grosor en la letra manuscrita
         """
         )
st.write("Esta es un red neuronal con la capacidad de clasificar el ancho de la letra manuscrita en imagenes ")
file = st.file_uploader("Por favor sube una imagen de letra manuscrita para ser clasificada", type=["jpg", "png"])



def import_and_predict(image_data, model):
    
        size = (120, 120)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(120, 120),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Por favor sube una imagen de letra Manuscrita para ser clasificada")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("Es Ancha!")
    else:
        st.write("Es Delgada!")
    
    st.text("Probability (0: Ancha, 1: Delgada)")
    st.write(prediction)
    #st.write(f"{prediction[0][np.argmax(prediction)]*100}%")
    #st.text(f"Precision: {(prediction[0][np.argmax(prediction)] * 100):.2f}%")