# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:13:37 2025

@author: polas
"""

import streamlit as st 
from PIL import Image

import numpy as np
#import cv2
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

model = load_model('best_modelqc3.h5')



background_image = """
<style>
[data-testid="stAppViewContainer"]  {
    background-image: url("https://img5.pic.in.th/file/secure-sv1/smsk-1e26f337bb6ec6813.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)




st.markdown("<h1 style='text-align: center; color: black ; font-size: 25px ;'>Sakhon QCX</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black ; font-size: 19px ;'><em>Good quality  Good using</em></h1>", unsafe_allow_html=True)
img_file = st.file_uploader("เปิดไฟล์ภาพ", type='jpg')



col1, col2 = st.columns([1,1]) 
#col3, col4 = st.columns([1,1]) 


if img_file is not None:
   
    im = Image.open(img_file)

    st.image(img_file,channels="BGR")
    
    img = Image.open(img_file)
    img = np.array(img)
    image = tf.cast(img, tf.float32) / 255.0
    image = tf.image.resize(image, (128, 128))
    X_submission = np.array(image)
    y = np.expand_dims(X_submission, 0)
    
    result = model.predict(y)
    
    Class_answer = np.argmax(result ,axis =1)
    if Class_answer == 0 :
        predict = 'Adequte'
    elif Class_answer == 1:
        predict = 'Anatomy'
    elif Class_answer == 2:
        predict = 'Rotation'
    elif Class_answer == 3:
        predict = 'Foreign Body'
    elif Class_answer == 4:
        predict = 'Open mouth'
    
    

    

    with col1:
        st.write("predict Quality of Water view")
    with col2:
        st.code(f'{predict}') 
    




