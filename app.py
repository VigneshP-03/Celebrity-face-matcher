import streamlit as st
from PIL import Image 
import numpy as np
import os
import cv2
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import pickle
from sklearn.metrics.pairwise import cosine_similarity

detector = MTCNN() # Creating an MTCNN object
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg') # Creating a VGGFace model with ResNet50 architecture
feature_list = pickle.load(open('embedding.pkl', 'rb')) # Loading the pickle file containing the list of features
filenames = pickle.load(open('filenames.pkl', 'rb')) # Loading the pickle file containing the list of filepaths

# Keeping the uploaded images history 
def save_uploaded_image(uploaded_image):
    try:
        with open(os.path.join('C:/Users/Vignesh/Desktop/Face-Matcher/venv/src/uploads', uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer()) 
        return True
    except:
        return False

def extract_features(img_path, model, detector):
    img = cv2.imread(img_path) # Reading the image
    results = detector.detect_faces(img) # Detecting the face in the image

    x,y,width,height = results[0]['box'] # Extracting the coordinates of the bounding box
    face = img[y:y+height, x:x+width] # Extracting the face from the image
    image = Image.fromarray(face) # Converting the image to PIL format
    image = image.resize((224,224)) # Resizing the image

    face_array = np.asarray(image) # Converting the image to an array
    face_array = face_array.astype('float32') # Converting the array to float32

    expanded_img = np.expand_dims(face_array, axis=0) # Expanding the dimensions of the image
    preprocessed_img = preprocess_input(expanded_img) # Preprocessing the image
    result = model.predict(preprocessed_img).flatten() # Extracting features from the image
    return result 

# Finding the most similar image
def recommend(feature_list, features):
    similarity = [] # Creating an empty list to store the similarity scores
    # Calculating the cosine similarity between the features of the uploaded image and the features of the images in the dataset 
    for i in range(len(feature_list)): 
        similarity.append(cosine_similarity(features.reshape(1,-1), feature_list[i].reshape(1,-1))[0][0]) 

    # Finding the index of the image with the highest similarity score enumerated in the similarity list
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos

st.title('Which celebrity are you?')
uploaded_image = st.file_uploader('Upload an image')

if uploaded_image is not None:
    # Saving the uploaded image
    if save_uploaded_image(uploaded_image):
        display_image = Image.open(uploaded_image) 
        features = extract_features(os.path.join('C:/Users/Vignesh/Desktop/Face-Matcher/venv/src/uploads', uploaded_image.name), model, detector) # Extracting features from the uploaded image and storing it in uploads folder
        index_pos = recommend(feature_list, features) # Finding the most similar image
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_')) # Extracting the name of the celebrity from the filepath
        col1,col2 = st.columns(2) # Creating two columns in the page

        with col1: # Displaying the uploaded image in the first column
            st.header('Your uploaded image')
            st.image(display_image)
        with col2: # Displaying the most similar image in the second column
            st.header('Looks like '+ predicted_actor) 
            st.image(filenames[index_pos], width = 300)