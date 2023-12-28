import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Qurollarni klassifikatsiya qiluvchi model")

file = st.file_uploader("Rasmni yuklash", type=['png', 'jpeg', 'svg', 'jfif'])

if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner('weapons_model.pkl')

    pred, pred_id , probs  = model.predict(img)

    st.success(f"Bashorat qiymat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)

    plt.figure()
    plt.bar(model.dls.vocab, probs)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Probability Distribution')

    st.pyplot(plt.gcf())

