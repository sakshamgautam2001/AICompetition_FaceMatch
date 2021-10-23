# Importing the Libs
import pickle
import pandas as pd
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
import cv2
from face_compare.model import facenet_model, img_to_encoding

# load model for extracting embeddings
model = facenet_model(input_shape=(3, 96, 96))

# Defining the function to take the image as input and returning the cropped image containing the face only
def get_face(img):
    '''Crops image to only include face plus a border'''
    height, width, _ = img.shape
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_box = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if(len(face_box) == 0):
        return img
    
    # Get dimensions of bounding box
    x = int(face_box[0][0])
    y = int(face_box[0][1])
    w = int(face_box[0][2])
    h = int(face_box[0][3])

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x+w)
    y2 = min(height, y+h)
    
    # Crop image
    cropped = img[y1:y2,x1:x2]
    return cropped

# Defining the function to get the embeddings from the cropped images containing faces, and concatenating the embeddings to get 256 sized vector for each pair
def run(image_one, image_two):
    # Load images
    face_one = get_face(cv2.imread(str(image_one), 1))
    face_two = get_face(cv2.imread(str(image_two), 1))

    # Calculate embedding vectors
    embedding_one = img_to_encoding(face_one, model)
    embedding_two = img_to_encoding(face_two, model)

    final_embedding = []
    for emb in embedding_one[0]:
        final_embedding.append(emb)
    for emb in embedding_two[0]:
        final_embedding.append(emb)
    return final_embedding


##########################################
# Loading the test file
test_images = pd.read_csv('test.csv')

# Creating the array to store the embeddings
X_test = []
for i in range(test_images.shape[0]):
    embedding = run("dataset_images/" + str(test_images['image1'][i]), "dataset_images/" + str(test_images['image2'][i]))
    X_test.append(embedding)
    print(i)

# Converting the array into numpy array
X_test = np.array(X_test)

# Importing the already trained Model
classifier = load_model('trained_model.h5')

# Creating the dataframe for results
results_df = pd.DataFrame()
results_df['image1'] = test_images['image1']
results_df['image2'] = test_images['image2']

# Predicting from the test embeddings
results_arr = classifier.predict(X_test)
results_arr = (results_arr > 0.5)

results_array = []
for r in results_arr:
    if(r == True):
        results_array.append(1)
    else:
        results_array.append(0)

results_df['results'] = results_array

# Saving the results into 'results.csv' file which will contain the results
results_df.to_csv('results.csv')

