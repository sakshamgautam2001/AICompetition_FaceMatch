# AICompetition_FaceMatch

**Installing the required packages**
The required packages are given in the requirements.txt file which needs to be installed to run the model

**Extracting out the Embeddings from the training model**
To train the model, first we need to extract out the embeddings associated with each pair in form of 256 sized vector.
To do that, run the "embedding.py" file, which will take train.csv as input and save the embedding arrays in form of pickle file

**Training the Model**
After the complete execution of "embedding.py" file, we will get the embeddings in form of "MainDataframe.pickle" file.
Now run the "model.py" file which will take "MainDataframe.pickle" embeddings as input and will train the model, the model after training will be saved as "trained_model.h5" file

**Predicting the Results**
After the model saved as "trained_model.h5" file, run the "predict.py" file, which will take "test.csv" and "trained_model.h5" as input, and will return the results in form of "results.csv" file

The model is already trained and saved as "trained_model.h5", so to predict the new set of pairs, the changes have to be made only in the "test.csv" file, and new results will be saved in "results.csv" file.

