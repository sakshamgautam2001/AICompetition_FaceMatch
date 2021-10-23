# Importing the Libraries
import cv2
import numpy as np
from face_compare.model import facenet_model, img_to_encoding
import pandas as pd
import pickle

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


##############################################

# Importing the training data
dataframe = pd.read_csv("train.csv")

# Creating arrays to store X and y data
X = []
y = []
for i in range(dataframe.shape[0]):
    embedding = run("dataset_images/" + str(dataframe['image1'][i]), "dataset_images/" + str(dataframe['image2'][i]))
    X.append(embedding)
    y.append(dataframe['label'][i])

# Saving the embeddings for training in form of Pickle file which will be further loaded to train the model
pickle.dump([X, y], open('DataframeTest.pickle', 'wb'))
print("Pickling done")





