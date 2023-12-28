import streamlit as st
from fastai.vision.all import *
import pathlib
import platform
import plotly.express as px
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Corrected platform check
if platform.system() == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath

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
    pred, pred_id, probs = model.predict(img)

    # Display prediction
    st.success(f"Bashorat qiymat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # Plot using Plotly
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)

    # Additional Matplotlib plot
    plt.figure()
    plt.bar(model.dls.vocab, probs)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Probability Distribution')

    # Display Matplotlib plot using st.pyplot()
    st.pyplot()

