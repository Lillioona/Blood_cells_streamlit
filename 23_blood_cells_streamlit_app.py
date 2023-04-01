#imports
import streamlit as st
import numpy as np
import pickle
import pandas as pd

from streamlit_option_menu import option_menu
from os import listdir     
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras.models import load_model

import requests
import base64

#------------------------------------------------------------------------------------------------------------------------------------------
# Overall page configuration
st.set_page_config(page_title="BCC", page_icon=":drop_of_blood:", layout="centered", initial_sidebar_state="auto", menu_items=None)

#------------------------------------------------------------------------------------------------------------------------------------------
#streamlit run "C:\Users\User\Desktop\streamlit\23_blood_cells_streamlit_app.py"

#------------------------------------------------------------------------------------------------------------------------------------------ 
# open images to display (maybe add them to corresponding section for clarity/structure
img_home_01 = Image.open('images/Blood_cell_examples.png')
img_EDA_01 = Image.open('images/Image_size.png')
img_EDA_02 = Image.open('images/EDA_02.png')
img_EDA_03 = Image.open('images/EDA_03.png')
Analysis_01 = Image.open('images/Analysis_01.png')
Analysis_02 = Image.open('images/Analysis_02.png')
Analysis_04_mix = Image.open('images/Analysis_04_mix.png')
Analysis_05_mix = Image.open('images/Analysis_05_mix.png')
#Analysis_06_mix = Image.open('images/Analysis_06_mix.png')
Analysis_07_Amri = Image.open('images/Analysis_07_Amri.png')
Analysis_08_Amri = Image.open('images/Analysis_08_Amri.png')
Analysis_09_Amri = Image.open('images/analysis_09_Amri.png')

#------------------------------------------------------------------------------------------------------------------------------------------
# Title of the Page
Header = st.container()
with Header:
    st.title('Automatic Blood Cell Recognition')

# Horizontal menu
selected = option_menu(None, ["Introduction", "E.D.A.", "Modelisation", 'Prediction', 'Perspectives', 'About'], 
    icons=["house-door", "bar-chart", "wrench", 'upload', 'search', 'info-circle'], 
    menu_icon="droplet", default_index=0, orientation="horizontal")

#------------------------------------------------------------------------------------------------------------------------------------------
# Section Home
if selected == 'Introduction':
    st.header('Introduction')
    
    # URL of the GIF on GitHub
    #url = "https://github.com/Lillioona/Blood_cells_streamlit/blob/main/red-blood-cells-national-geographic.gif"

    file_ = open("red-blood-cells-national-geographic.gif", "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
    f'<div style="text-align:center;"><img src="data:image/gif;base64,{data_url}" alt="cat gif"></div>',
    unsafe_allow_html=True,
    )
    
    st.markdown(''' Blood is a body fluid which flows in the human circulation system and has important functions, such as the supplement of necessary 
    substances such as nutrients and oxygen to cells, removing waste and immune defense. 

    By the change of their blood components in blood count many diseases can be discovered as well as their severity, 
    because of that blood is one of the most examined body fluid in the medical laboratory. 

    Especially for hematological diseases, the analysis of the morphology of blood is well known and used in form of blood smear review.
    However, to detect morphological differences between distinct types of normal and abnormal peripheral blood cells, it requires experience, 
    skills and time.
    Therefore, it is very helpful for hematological diagnosis the use of automatic blood cell recognition system.

    The main object of this project is to develop a deep learning models to recognize different types of blood cells.
    In general blood cells can be divided into erythrocytes known as red blood cells , leukocytes known as white blood cells and the cell fragments 
    called platelets or thrombocytes.
    In this study the focus lies on erythroblasts which are an early stage of erythrocytes and the subdivision of leukocytes such as neutrophils,
    basophils, eosinophils, monocytes ,lymphocytes and immature granulocytes(IG) and the as mentioned above, platelets.''')
    
    st.image(img_home_01, caption = 'different types of blood cells', align="center")
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("*The data which enabled this project was derived from three different sources. The entire data is publicly available:*")
    
    st.markdown("""<div style="color:#696969">
    <ul>
        <li><b>Barcelona:</b> A dataset of microscopic peripheral blood cell images for development of automatic recognition systems, 2020 - 
            <a href="https://data.mendeley.com/datasets/snkd93bnjr/1">https://data.mendeley.com/datasets/snkd93bnjr/1</a></li>
        <li><b>Munich:</b> A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Contols (AML-Cytomorhology LMU), 2022 - 
            <a href="https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/77?passcode=a6be8bf0a97ddb34fc0913f37b8180d8f7d616a7">
            https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/77?passcode=a6be8bf0a97ddb34fc0913f37b8180d8f7d616a7</a></li>
        <li><b>Rabbin:</b> A large dataset of white blood cells containing cell locations and types, along with segmented nuclei and cytoplasm, 2022 - 
            <a href="http://dl.raabindata.com/Leukemia_Data/ALL/L1/">http://dl.raabindata.com/Leukemia_Data/ALL/L1/</a></li>
    </ul>
    </div>""", unsafe_allow_html=True)


#------------------------------------------------------------------------------------------------------------------------------------------
#Section EDA    
if selected == 'E.D.A.':
    st.header('Exploratory Data Analysis')
    
    df = pd.read_csv("dataframe_eda.csv", index_col=0)
    st.dataframe(df)
    
    
    st.markdown('''text text text text text text text text text text text text text text text text text text text text text text text
                text text text text text text text text text text text text text text text text text text text text text text text text
                text text text text text text text text text text text text text''')
    st.image(img_EDA_01, caption = 'img title')
    st.image(img_EDA_02, caption = 'img title')    
    st.image(img_EDA_03, caption = 'img title')
    
#------------------------------------------------------------------------------------------------------------------------------------------    
#Section Models     
if selected == 'Modelisation':
    st.header('Modelisation')
    st.markdown('In the following we present the models obtaining the best prediction results:')
    st.subheader('ResNet50V2 as base model')
    
    st.markdown(
        """
        Simple model:
        - Image augmentation: horizontal & vertical flips, random rotations and center crop augmentation
        - Layer architecture: global average pooling layer, no dropout layers, finishing with a flattened layer and a dense layer with
        a high number of units (before the output layer) 
        - F1-score: 91%
        """)
    
    st.markdown(
        """
        With fine-tuning:
        - the last (5th) Conv-block set to be trainable 
        - this resulted in over 15 million trainable parameters compared to the initial 164.568 parameters
        - F1-score: 98%
        """) 
    
    col1, col2 = st.columns(2)
    col1.image(Analysis_01, use_column_width=True, caption = 'ResNet50V2 Loss')
    col2.image(Analysis_02, use_column_width=True, caption = 'ResNet50V2 Accuracy')
    
    st.markdown(
        """
        Mixed inputs: 
        - The features luminosity and brightness were used as numerical input 
        - next to the image arrays
        - same architecture and fine-tuning as the previous model
        - F1-score: 97.3%
        """)
    
    col1, col2 = st.columns(2)
    col1.image(Analysis_04_mix, use_column_width=True, caption = 'Mixed Inputs Loss')
    col2.image(Analysis_05_mix, use_column_width=True, caption = 'Mixed Inputs Accuracy')
    
    # st.image(Analysis_06_mix, caption = 'Confusion Matrix')
    
    st.subheader('VGG16 as base model')
    st.markdown(
        """
        Simple model:
        - Image: architecture:
        - Layer architecture: global average pooling layer, two large Dense layers followed by a slight dropout layer
        - F1-score: 86%
        """)
    st.markdown(
        """
        With fine-tuning: 
        - the last 3 layers were set to trainable 
        - this resulted in close to six million trainable parameters compared to the initial 1.054.216 parameters.
        - F1-score: 96%
        """)

    col1, col2 = st.columns(2)
    col1.image(Analysis_07_Amri, use_column_width=True, caption = 'VGG16 Loss')
    col2.image(Analysis_08_Amri, use_column_width=True, caption = 'VGG16 Accuracy')

    st.image(Analysis_09_Amri, caption = 'VGG16 Confusion Matrix')
#------------------------------------------------------------------------------------------------------------------------------------------   
#Section Prediction    
if selected == 'Prediction':
    st.header('Prediction')
    st.subheader("Choose the model for prediction")
   
       

           
          #  col1, col2 = st.columns(2)
            # load dataset 1
       #     with col1:
       #         file = st.file_uploader(label='Pick an image to test',accept_multiple_files=False)
        #        prediction(file)
            # load dataset 2
        #    with col2:
        #        st.write("Select images")


# function for a model to predict blood cell from an image
def prediction(file):
    if file is not None:
            image_data = Image.open(file)            
            st.image(image_data, width=180)
 
            size = (360,360)    
            image = ImageOps.fit(image_data, size, Image.BICUBIC)
            image = np.asarray(image)

            img = image[:, :, ::-1]
            #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255

            img_reshape = img[np.newaxis,...]
            prediction = model.predict(img_reshape)

            predicted_class = np.argmax(prediction)
            #st.write(predicted_class)
            true_classes_list = ['basophil',
                                        'eosinophil',
                                        'erythroblast',
                                        'ig',
                                        'lymphocyte',
                                        'monocyte',
                                        'neutrophil',
                                        'platelet']
            st.write(f'This image most likely belongs to {true_classes_list[predicted_class]} with a probability of {prediction.max()}%')   

# list all available images to make predicitions on    
def list_images(directory, file_type):
    directory += file_type
    files = listdir(directory)
    files[0] = "Select from list"
    file = st.selectbox("Pick an image to test",files) 
    return file

# calculate f1 score
def f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall       

#------------------------------------------------------------------------------------------------------------------------------------------
#Section Perspectives    
if selected == 'Perspectives':
    st.header('Perspectives')
    st.markdown("""
        The role of machine learning methods in intelligent medical diagnostics is becoming more and more present these days.
        And deep neural networks are revolutionizing the medical diagnosis process rapidly.

         - in reality,there is a wider spectrum of blood cell types, regarding to subcategories of immature granulocytes and other early stages of blood cell.
         In this project the focus was to detect 8 different blood cells types.

        - the training data set should be as diverse and precise as possible to classify the blood cells.

        - different sources can considerly change the outcome of images, like the different medical devices, microscope and camera,
        and the method of processing the blood cells, the use of stain.

        - the dataset can be used to recognize the blood cell type and trained further to classify other types of abnormal cells.
        """)
#------------------------------------------------------------------------------------------------------------------------------------------    
#Section About    
if selected == 'About':
    st.header('About')
    st.markdown('This machine learning project was part of Datascientest International Class at University of Paris La Sorbonne.')
    st.header('Contributors')
    st.write('Amritha Kalluvettukuzhiyil  \n Elias Zitterbarth  \n Daniela Hummel  \n Lilli Krizek')
