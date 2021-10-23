# Importing Libs
import pickle
import pandas as pd
import numpy as np
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout

# Loading the Pickle file that was saved from the embedding.py file
Xydata = pickle.load(open('MainDataframe.pickle', 'rb'))
# Sorting the X and y data
X = Xydata[0]
y = Xydata[1]

X = np.array(X)
y = np.array(y)

# Building Classifier Model
classifier = Sequential()
classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu', input_dim = 256))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.25))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Splitting the model into training and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 48)

# Fitting the classifier to train the model
classifier.fit(X_train, y_train, batch_size = 64, epochs = 150, validation_data = (X_test, y_test))
classifier.save("trained_model.h5")     # Saving the trained model for making further predictions







