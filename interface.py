import streamlit as st
import re
import pandas as pd 
import numpy as np

st.title("""
form input data, by Izul Ramdani
""")

deskripsi, import_data, preprocessing, modelling, implementation = st.tabs(["Description", "Import Data", "Preprocessing", "Modelling", "Implementation"])

#Fractional Knapsack Problem
#Getting input from user
ph=int(st.number_input("Masukan pH: ",0))
temperatur=int(st.number_input("Masukkan Temperatur : ",0))
taste=int(st.number_input("Masukkan Taste : ",0))
odor=int(st.number_input("Masukkan Odor : ",0))
fat=int(st.number_input("Masukkan Fat : ",0))
turbidity=int(st.number_input("Masukkan Turbidity : ",0))
colour=int(st.number_input("Masukkan Colour : ",0))

submit = st.button("submit")


if submit:
    st.info("Jadi,Grade nya adalah . ")


