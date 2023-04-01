#imports
import streamlit as st
import numpy as np
import pickle
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns

from streamlit_option_menu import option_menu
from os import listdir     
from PIL import Image, ImageOps
from io import BytesIO

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
img_EDA_02 = Image.open('images/RGB_dist.png')
img_EDA_03 = Image.open('images/Grey_dist.png')
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
    
    st.write("\n\n")
    
    
    st.markdown(''' 
    Blood is a body fluid which flows in the human circulation system and has important functions, such as the supplement of necessary 
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
    basophils, eosinophils, monocytes ,lymphocytes and immature granulocytes(IG) and the as mentioned above, platelets.\n''')
    
    # image blood cells
    st.image(img_home_01, caption = 'The different types of blood cells to classify')
    
    #horizontal line
    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.markdown("*The data which enabled this project was derived from three different sources. The entire data is publicly available:*")
    
    st.markdown("""<div style="color:#696969">
    <ul>
        <li><b>Barcelona:</b> A dataset of microscopic peripheral blood cell images for development of automatic recognition systems, 2020 - 
            <a href="https://data.mendeley.com/datasets/snkd93bnjr/1">https://data.mendeley.com/datasets/snkd93bnjr/1</a></li>
        <li><b>Munich:</b> A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Contols (AML-Cytomorhology LMU), 2022 - 
            <a href="https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/77?passcode=a6be8bf0a97ddb34fc0913f37b8180d8f7d616a7">
            https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/77?passcode=a6be8bf0a97ddb34fc0913f37b8180d8f7d616a7</a></li>
        <li><b>Raabin:</b> A large dataset of white blood cells containing cell locations and types, along with segmented nuclei and cytoplasm, 2022 - 
            <a href="http://dl.raabindata.com/Leukemia_Data/ALL/L1/">http://dl.raabindata.com/Leukemia_Data/ALL/L1/</a></li>
    </ul>
    </div>""", unsafe_allow_html=True)


#------------------------------------------------------------------------------------------------------------------------------------------
#Section EDA    
if selected == 'E.D.A.':
    st.header('Exploratory Data Analysis')
    st.markdown(
        """
        Three open source datasets were used to achieve this project's objective. In total they contained ~52,000 images of blood cells. The classification into one of 
        eight blood cell types was the target of this project’s model.
        """
    )
    with st.expander("Further information on the datasets"): 
        st.subheader('Barcelona')
        st.markdown(
            """
            The first dataset was acquired using the analyzer CellaVision DM96 in the Core Laboratory at the Hospital Clinic of Barcelona. It is organized into eight 
            different groups: neutrophils, eosinophils, basophils, lymphocytes, monocytes, immature granulocytes (IG; includung promyelocytes, myelocytes, and metamyelocytes), 
            erythroblasts and platelets or thrombocytes. The original image size was 360 × 363 pixels, in format .jpg, and they were annotated by expert clinical pathologists. 
            The images were captured from individuals without infection, hematologic or oncologic disease and free of any pharmacologic treatment at the time of blood collection.
            """
        )
        st.subheader('Raabin')
        st.markdown(
            """
            The second dataset is called Raabin-WBC. It is a collection of images from the Razi Hospital in Rasht, Gholhak Laboratory, Shahr-e-Qods Laboratory, and 
            Takht-e Tavous Laboratory in Tehran.
            """
        )
        st.subheader('Munich')
        st.markdown(
            """
            The third was the Munich AML Morphology dataset which contained images taken from peripheral blood smears of 100 patients diagnosed with Acute Myeloid Leukemia 
            at Munich University Hospital between 2014 and 2017, as well as 100 patients without signs of hematological malignancy with an M8 digital microscope.
            """
        )
        st.subheader('Classes')
        st.markdown(
            """
            Depending on the source, the number of different blood cell classes varied between 8 to 13 classes. Based on the provided descriptions for each dataset and their 
            classes, it was decided to merge them into a total of the 8 classes as described for the Barcelona dataset: Neutrophils, eosinophils, basophils, lymphocytes, 
            monocytes, immature granulocytes (promyelocytes, myelocytes, and metamyelocytes), erythroblasts and platelets or thrombocytes.
            """
        )
        
    st.markdown(
    """
    An extract of the combined dataset with additional features gathered from the files:
    """
    )
    df = pd.read_csv("dataframe_eda.csv", index_col=0)
    st.dataframe(df)
    
    st.subheader('Distribution of the image sizes')
    st.markdown("""
    The image size varied between the different datasets, as displayed in the following scatter plot. They were all resized to 360x360 for the 
                continuous process.
                """)
    st.image(img_EDA_01)
    
    # set plot style
    #sns.set_style('darkgrid')

    # create plot
    #figure, axis = plt.subplots(figsize=(8,8), facecolor="#0e1117")
    #axis.set_facecolor("#0e1117")
    #axis.set_xlim([351,599])
    #axis.set_ylim([351,599])
    #axis.set_xlabel('Width', color='white')
    #axis.set_ylabel('Height', color='white')
    #axis.spines['top'].set_visible(False)
    #axis.spines['right'].set_visible(False)
    #axis.set_title("Original image resolution", fontdict={'color': "white"}, size= 18, pad=25)
    #axis.tick_params(colors="white", bottom=True, left=True)
    #ax = sns.scatterplot(data=df, x='width', y='height',  hue='origin',
    #                    palette='deep', size= 'origin', sizes=(100, 200))
    #axis.legend(loc=(0.125,.82), frameon=True, fontsize="large")

    # display plot in Streamlit
    #st.pyplot(figure)
    
    
    st.subheader('Brightness')
    st.markdown("""
    The brightness is calculated by the RGB pixel distribution, which show different characteristics according to the classes. 
    """)
    st.image(img_EDA_02, caption = 'RGB pixel distribution of the images per class') 
    st.subheader('Luminance')
    st.markdown("""
    The luminance is calculated by the greyscale pixel distribution.
                """)
    st.image(img_EDA_03, caption = 'Greyscale pixel distribution of the images per class')
    
#------------------------------------------------------------------------------------------------------------------------------------------    
#Section Models     
if selected == 'Modelisation':
    st.header('Modelisation')
    st.markdown('In the following we present the models that achieved the best prediction results:')         
    
    with st.expander("Further information about the modelisation process"): 
        st.subheader('First steps')
        st.markdown(
            """
            We started with four pre-trained models ResNet50V2, VGG16, MobileNetV2 and Xception. Without notable image preprocessing or modification 
            of layers, hyper-parameters and an imbalanced dataset, the resulting accuracies remained close to random (~12,5% F1). We also faced memory 
            issues while working with the entire dataset of ~52.000 images. 
            """
        )
        st.subheader('Subsample')
        st.markdown(
            """
            A subsample was created to solve imbalance and memory issues. Based on the class with the smallest representation (Basophil, n=1598), a total 
            number of 12784 images was extracted, with each class being evenly represented. This was done using pandas' groupby and sample method. The 
            subsample was distributed to every member of the group to stay comparable during modelisation.
            """
        ) 
        st.subheader('Image Augmentation')
        st.markdown(
            """
            Image augmentation can be usefull to train classification models with small datasets and reduce overfitting. In this case it didn’t. The 
            classical ImageDataGenerators resulted in continuously higher validation scores compared to training scores, as well as 
            longer runtimes. Considering that blood cell images tend to be recorded in standardized environments with similar methodologies, it was hypothesized 
            that too much data augmentation would actually decrease the model's performance. Reducing the image augmentation to horizontal & vertical flips, as well 
            as adding random rotations thorugh an augmentation layer combined with re-thinking the layer architecture resulted in the first model hitting above 
            an 80% F1-score. Regarding the fact, that the most important information of the image (the white blood cell to classify) tends to be in the center of the 
            image, surrounded by non-essential red blood cells, it was hypothesized that center crop augmentation would be beneficial. It increased the initial F1 score 
            to ~88%.
            """
        )
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

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Necessary function and variables
#MODEL_URL = "https://www.dropbox.com/s/hztf508d1b44ywu/Best_model_ft_5th_layer.h5?dl=1"
MODEL = "Best_model_ft_5th_layer.h5"

IMG_SIZE = (360,360) 

CLASS_LABELS = ['basophil',
                'eosinophil',
                'erythroblast',
                'ig',
                'lymphocyte',
                'monocyte',
                'neutrophil',
                'platelet']

@st.cache(allow_output_mutation=True)
def load_model():
    #model_file = BytesIO(requests.get(MODEL_URL).content)
    model = tf.keras.models.load_model(MODEL)
    return model

# Calculate f1 score
def f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall

#load the model to use for predictions
try:
    #Create a dictionary mapping the function name to the function object
    custom_objects = {'f1': f1}

    # Load the Keras model using custom_object_scope
    with tf.keras.utils.custom_object_scope(custom_objects):
        model = load_model()

except Exception as e:
    st.write(e)

# Preprocess image
def preprocess_image(image):
    if file is not None:
        image = image.resize(IMG_SIZE)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image
    
# Function to make predictions
def predict(image):
    if file is not None:
        image = preprocess_image(image)
        predictions = model.predict(tf.expand_dims(image, axis=0))[0]
        predicted_class = CLASS_LABELS[predictions.argmax()]
        confidence = predictions.max()
        return predicted_class, confidence
    
  
          #  col1, col2 = st.columns(2)
            # load dataset 1
       #     with col1:
       #         file = st.file_uploader(label='Pick an image to test',accept_multiple_files=False)
        #        prediction(file)
            # load dataset 2
        #    with col2:
        #        st.write("Select images")

        
          

# list all available images to make predicitions on (no images uploaded so far right?)   
def list_images(directory, file_type):
    directory += file_type
    files = listdir(directory)
    files[0] = "Select from list"
    file = st.selectbox("Pick an image to test",files) 
    return file

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
if selected == 'Prediction':
    st.header('Prediction')
    st.subheader("Choose a model to classify a blood cell image")
    
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        predicted_class, confidence = predict(image)
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")
        
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
