
import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

lr = data["model"]


def show_predict_page():
    st.title("Sentiment prediction")

    text_input = st.text_input(
        "Enter some text"
    )

    ok = st.button("Enter")
    if ok:
        X = [text_input]

        sentiment = lr.predict(X)
        
        if sentiment[0] == 1:
            ans = "positive"
        elif sentiment[0] == 0:
            ans = "negative"

        st.subheader(f"The sentiment is {ans}")