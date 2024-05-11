import streamlit as st
from fastai.vision.all import *
import pathlib
import platform
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

# Corrected platform check
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

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

    # Additional Matplotlib plot: Probability Distribution
    plt.figure()
    plt.bar(model.dls.vocab, probs)
    plt.xlabel('Sinf')
    plt.ylabel('Ehtimolligi')
    plt.xticks(rotation=90)
    plt.title('Ehtimollik taqsimoti')

    # Display Matplotlib plot using st.pyplot()
    st.pyplot()

    # Compute confusion matrix
    interp = ClassificationInterpretation.from_learner(model)
    _, _, cm = interp.confusion_matrix()

    # Plot confusion matrix
    st.subheader("Confusion Matrix")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.dls.vocab, yticklabels=model.dls.vocab)
    plt.xlabel("Bashorat qiymat")
    plt.ylabel("Haqiqiy qiymat")
    st.pyplot()

    # Distribution of Probabilities
    st.subheader("Ehtimollik taqsimoti")
    plt.figure()
    sns.histplot(probs, bins=10, kde=True)
    plt.xlabel("Ehtimolligi")
    plt.ylabel("Aniqligi")
    st.pyplot()
