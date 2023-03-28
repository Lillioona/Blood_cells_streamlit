import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image

#streamlit run "C:\Users\User\Desktop\streamlit\23_blood_cells_streamlit_app.py"

#images
img_home_01 = Image.open('images/cell_images.png')
img_EDA_01 = Image.open('images/EDA_01.png')
img_EDA_02 = Image.open('images/EDA_02.png')
img_EDA_03 = Image.open('images/EDA_03.png')

#Title of the Page
Header = st.container()
with Header:
    st.title('Automatic Blood Cell Recognition')


# horizontal menu
selected = option_menu(None, ["Introduction", "E.D.A.", "Modelisation", 'Prediction', 'Perspectives', 'About'], 
    icons=["house-door", "droplet", "droplet", 'droplet', 'droplet', 'envelope'], 
    menu_icon="droplet", default_index=0, orientation="horizontal")

#Section Home
if selected == 'Introduction':
    st.header('Introduction')
    st.markdown('text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text')
    st.image(img_home_01, caption = 'img title')
    
#Section EDA    
if selected == 'E.D.A.':
    st.header('Exploratory Data Analysis')
    st.markdown('text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text')
    st.image(img_EDA_01, caption = 'img title')
    st.image(img_EDA_02, caption = 'img title')    
    st.image(img_EDA_03, caption = 'img title')  
    
#Section Models     
if selected == 'Modelisation':
  #  st.header('Modelisation, first steps')
  #  st.markdown('We started with four pretrained models ResNet50V2, VGG16, MobileNetV2 and Xception. Without notable image preprocessing, modification of layers or hyper parameters and the imbalanced dataset the resulting accuracies remained close to random (~12,5% F1). Also we faced memory errors working with the whole dataset of 52 000 images. ')
  #  st.header('Subsample')
  #  st.markdown('To solve imbalance and memory issues a subsample was created. Regarding the class with the smallest occurrence (Basophil, n = 1598) a total number of 12784 images was extracted, where now every class was evenly represented. This was done using pandas methods groupby and sample. The subsample was given to every member of the group to stay comparable in modelisation.')
  #  st.header('Image Augmentation')
  #  st.markdown('Image augmentation can be usefull to train your model and prevent it from overfitting, but in this case it didn’t. The classical ImageDataGenerators resulted in continuously higher validation scores compared to training scores and long runtimes varying between a couple of hours up to an entire day to train a single model. Considering that blood cell images tend to be recorded in standardized environments with similar methodologies, it was hypothesized that too much data augmentation was decreasing performance of the model. Reducing the image augmentation to horizontal & vertical flips, as well as random rotations in the form of an augmentation layer combined with rethinking the layer architecture resulted in the first model hitting above an 80% F1 score. Regarding that the most important information of the image (the white blood cell to classify) tended to be in the center of the image, surrounded by non-essential red blood cells, it was hypothesized that center crop augmentation would be beneficial. It increased the F1 score to ~88%.')
  #  st.header('Optimizations')
    
#Section Prediction    
if selected == 'Prediction':
    st.header('Prediction')
    st.markdown('text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text')

#Section Perspectives    
if selected == 'Perspectives':
    st.header('Perspectives')
    st.markdown('text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text text')

#Section About    
if selected == 'About':
    st.header('About')
    st.markdown('This mashine learning project was part of Datascientest International Class at University of Paris La Sorbonne.')
    st.header('Contributors')
    st.write('Amritha Prasneh  \n Elias Zitterbarth  \n Daniela Hummel  \n Lilli Krizek')
  
