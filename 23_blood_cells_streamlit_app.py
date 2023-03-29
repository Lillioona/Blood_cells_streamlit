import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from os import listdir
#import cv2
import numpy as np
from PIL import Image, ImageOps
#import tensorflow as tf
#from tensorflow.keras.models import load_model

"""
def prediction(file):
    if file is not None:
            image_data = Image.open(file)
            st.image(image_data, width=180)
 
            size = (360,360)    
            image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
            image = np.asarray(image)

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
            st.write('This image most likely belongs to ', true_classes_list[predicted_class])

    
def list_images(directory, file_type):
    directory += file_type
    files = listdir(directory)
    files[0] = "Select from list"
    file = st.selectbox("Pick an image to test",files) 
    return file
        
def f1(y_true, y_pred):    
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        
        recall = TP / (Positives+K.epsilon())    
        return recall     
"""
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
    st.header('Modelisation')
  #  st.markdown('We started with four pretrained models ResNet50V2, VGG16, MobileNetV2 and Xception. Without notable image preprocessing, modification of layers or hyper parameters and the imbalanced dataset the resulting accuracies remained close to random (~12,5% F1). Also we faced memory errors working with the whole dataset of 52 000 images. ')
  #  st.header('Subsample')
  #  st.markdown('To solve imbalance and memory issues a subsample was created. Regarding the class with the smallest occurrence (Basophil, n = 1598) a total number of 12784 images was extracted, where now every class was evenly represented. This was done using pandas methods groupby and sample. The subsample was given to every member of the group to stay comparable in modelisation.')
  #  st.header('Image Augmentation')
  #  st.markdown('Image augmentation can be usefull to train your model and prevent it from overfitting, but in this case it didn’t. The classical ImageDataGenerators resulted in continuously higher validation scores compared to training scores and long runtimes varying between a couple of hours up to an entire day to train a single model. Considering that blood cell images tend to be recorded in standardized environments with similar methodologies, it was hypothesized that too much data augmentation was decreasing performance of the model. Reducing the image augmentation to horizontal & vertical flips, as well as random rotations in the form of an augmentation layer combined with rethinking the layer architecture resulted in the first model hitting above an 80% F1 score. Regarding that the most important information of the image (the white blood cell to classify) tended to be in the center of the image, surrounded by non-essential red blood cells, it was hypothesized that center crop augmentation would be beneficial. It increased the F1 score to ~88%.')
  #  st.header('Optimizations')
    
#Section Prediction    
if selected == 'Prediction':
    st.header('Prediction')
    """
    st.subheader("Choose the model for prediction")
try:
    model = st.selectbox("XXX", 
                        ["Select one model from the list", "ResNet", "ResNet_MixedInput", "VGG16"],
                        label_visibility = "hidden")
    
    if not model:
        st.error("Please select at least one model.")
        
    elif(model != "Select one model from the list"):
        st.subheader("Select the image for prediction")

        # load model
        if(model == 'ResNet'):
            model = load_model('models/Best_model_ft_5th_layer.h5', custom_objects={'f1':f1})
           # model = load_model('/Users/prasnehpuzhakkal/Downloads/Best_model_ft_5th_layer.h5', 
                              # custom_objects={'f1':f1})
        elif(model =='ResNet_MixedInput'):
            model = load_model('models/final_mixed_input_model_ft_no_bpc.h5', custom_objects={'f1':f1})
            #model = load_model('/Users/prasnehpuzhakkal/Downloads/final_mixed_input_model_ft_no_bpc.h5', 
                              # custom_objects={'f1':f1}) 
        elif(model =='VGG16'):
            model = load_model('models/vgg16_augmented_model.h5')
            #model = load_model('/Users/prasnehpuzhakkal/Downloads/vgg16_augmented_model.h5')
            
        #placeholder = st.empty()
        #with placeholder.container():

        col1, col2 = st.columns(2)

        # load dataset 1
        with col1:
            file = st.file_uploader(label='Pick an image to test',
                                                accept_multiple_files=False)
            prediction(file)

        # load dataset 2
        
        with col2:
            directory = '/Volumes/WDElements/Amritha/XXX/'
            file_type = st.selectbox("Select your favorite image type",
                                    ('basophil','eosinophil','erythroblast','ig','lymphocyte','monocyte','neutrophil','platelet'),
                                    )
            file = list_images(directory, file_type)
            file_path = directory+file_type+'/'+file
            if(file != "Select from list"):
                prediction(file_path)
         
                                                   
        except:
            st.error("error")
            """
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
  
