import streamlit as st
import pandas as pd
import numpy as np
import matplotlib as plt


st.title("My First Streamlit App 🚀")
st.write("Hello World, welcome to Streamlit!")

import streamlit as st

st.title("About Me")
st.write("Hi, I'm Chhavi!")
st.markdown("I like tech + writing")
st.write("This is a simple streamlit tutorial")
st.button("click me")
checker = st.checkbox("do you know me? tick if yes!")
st.write("below is a useless message")
if checker:
    st.write("checkbox is checkedd damn")
option=st.radio("choose an option", ["first", "second", "third"])

mood = st.selectbox("How are you?", ["Happy", "Sad", "Tired", "Angry"])
if mood == "Happy":
    st.write("Nice 😄")
elif mood == "Sad":
    st.write("Don't Be")
elif mood == "Tired":
    st.write("Take rest")
elif mood == "Angry":
    st.write("Being angry is bad :(")

