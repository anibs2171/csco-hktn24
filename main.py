import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="My App",layout='wide')

st.markdown("# IIOT hackathon")

st.text('Time series analytics for your equipment')

if st.button('Try it Out !'):
    switch_page("Analytics")

