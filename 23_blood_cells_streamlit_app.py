import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

#streamlit run "C:\Users\User\Desktop\streamlit\23_blood_cells_streamlit_app.py"

#images
#img_home_01 = Image.open('C:/Users/User/Desktop/streamlit/images/cell_images.png')
#img_EDA_01 = Image.open('C:/Users/User/Desktop/streamlit/images/EDA_01.png')
#img_EDA_02 = Image.open('C:/Users/User/Desktop/streamlit/images/EDA_02.png')
#img_EDA_03 = Image.open('C:/Users/User/Desktop/streamlit/images/EDA_03.png')

Header = st.container()


with Header:
    st.title('Automatic Blood Cell Recognition')


# horizontal menu
selected = option_menu(None, ["Home", "EDA", "Models", 'Analysis', 'Prediction', 'Perspectives', 'About'], 
    icons=["house-door", "droplet", "droplet", 'droplet', 'droplet', 'droplet', 'envelope'], 
    menu_icon="droplet", default_index=0, orientation="horizontal")


if selected == 'Home':
    st.header('Introduction')
    st.markdown('text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text')
#    st.image(img_home_01, caption = 'img title')
    
    
if selected == 'EDA':
    st.header('Exploratory Data Analysis')
    st.markdown('text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text')
 #   st.image(img_EDA_01, caption = 'img title')
 #   st.image(img_EDA_02, caption = 'img title')    
 #   st.image(img_EDA_03, caption = 'img title')  
    
    
if selected == 'Models':
    st.header('Models')
    st.markdown('text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text')

if selected == 'Analysis':
    st.header('Analysis')
    st.markdown('text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text')

if selected == 'Prediction':
    st.header('Prediction')
    st.markdown('text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text')

if selected == 'Perspectives':
    st.header('Perspectives')
    st.markdown('text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text')
       
if selected == 'About':
    st.header('About')
    st.markdown('text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text')    
