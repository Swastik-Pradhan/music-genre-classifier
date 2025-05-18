# test_app.py
import streamlit as st

st.title("Hello Streamlit!")
st.write("If you see this, Streamlit is working locally.")
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")