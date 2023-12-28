import streamlit as st
from fastai.vision.all import *
import pathlib
import platform
import plotly.express as px
import matplotlib.pyplot as plt

# Disable Matplotlib global use warning
st.set_option('deprecation.showPyplotGlobalUse', False)

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# title
st.title("Qurollarni klassifikatsiya qiluvchi model")

# rasmni joylash
file = st.file_uploader("Rasmni yuklash", type=['png', 'jpeg', 'svg', 'jfif'])

# Check if a file is uploaded
if file:
    st.image(file)

    # PIL convert
    img = PILImage.create(file)

    # model
    model = load_learner('weapons_model.pkl')

    # Bashorat
    pred, pred_id , probs  = model.predict(img)

    # Display prediction
    st.success(f"Bashorat qiymat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # Plot using Plotly
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
