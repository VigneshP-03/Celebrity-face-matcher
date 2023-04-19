import os 
import numpy as np 
import pickle # For saving the features
from tqdm import tqdm # For progress bar

actors = os.listdir('data') # List of folders inside the data folder
print(actors) # Displaying the folder names 
filenames=[] 

# Inserting filepaths inside the data folder into a list 
for actor in actors:
    for file in os.listdir(os.path.join('data',actor)):
        filenames.append(os.path.join('data',actor,file))

# Dumping the list into a pickle file 
pickle.dump(filenames, open('filenames.pkl', 'wb'))

from tensorflow.keras.preprocessing import image 
from keras_vggface.utils import preprocess_input 
from keras_vggface.vggface import VGGFace 
from keras.utils.layer_utils import get_source_inputs

# Loading the pickle file containing the list of filepaths
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Creating a VGGFace model with ResNet50 architecture
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Extracting features from the images
def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224)) # Loading the image
    img_array = image.img_to_array(img) # Converting the image to an array
    expanded_img = np.expand_dims(img_array, axis=0) # Expanding the dimensions of the image
    preprocessed_img = preprocess_input(expanded_img) # Preprocessing the image
    result = model.predict(preprocessed_img).flatten() # Extracting features from the image
    return result 

features = [] # Creating an empty list to store the features
for file in tqdm(filenames): # Looping through the list of filepaths
    features.append(feature_extractor(file, model)) # Inserting the extracted features to the list

pickle.dump(features, open('embedding.pkl', 'wb')) # Dumping the features into a pickle file 
