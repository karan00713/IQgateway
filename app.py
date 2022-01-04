import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title("Malaria Detection")
st.text("Enter the URL of the Image for Detection")

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('C:/Users/KARAN K S/model.h5')
  return model

with st.spinner('Loading Model Into Memory....'):
  model=load_model()

classes=['infected','uninfected']

def scale(image):
  image = tf.cast(image,tf.float32)
  image /= 255.0
  return tf.image.resize(image,[224,224])

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels=3)
  img = scale(img)
  return np.expand_dims(img, axis=0)

path = st.text_input('Enter Image URL:','https://miro.medium.com/max/508/0*sCZHuHECn0zkH3fR.png')
if path is not None:
  content = requests.get(path).content

  st.write("Predicted Class:")
  with st.spinner('classifying....'):
    label = np.argmax(model.predict(decode_img(content)),axis=1)
    st.write(classes[label[0]])
  st.write("")
  image = Image.open(BytesIO(content))
  st.image(image, caption='Detecting Malaria', use_column_width=True)