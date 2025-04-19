import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import streamlit as st
from PIL import Image


# loading the csv data to a Pandas DataFrame
heart_data_table = pd.read_csv('heart_disease_data.csv')
heart_data_table.head()
X = heart_data_table.drop(columns='target', axis=1)
Y = heart_data_table['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
# using logic regrerssion
my_model = LogisticRegression()

# Training this LogisticRegression model with Training data
my_model.fit(X_train, Y_train)

# getting/finding the accuracy on training data
X_train_prediction = my_model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
# getting/finding the accuracy on test data
X_test_prediction = my_model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# web app
st.title('Heart Disease Preidction Model')

input_text = st.text_input('Provide comma separated features to predict heart disease, in the input space')
sprted_input = input_text.split(',')

# You need to have the image in this project folder
img = Image.open('heart.jpg')
#setting the size of the image
st.image(img,width=200)

try:
    np_df = np.asarray(sprted_input,dtype=float)
    reshaped_df = np_df.reshape(1,-1)
    prediction = my_model.predict(reshaped_df)
    if prediction[0] == 0:
        st.write("Output: This person doesn't have heart disease")
    else:
        st.write("Output: This person has heart disease")

except ValueError:
    #
    st.write('Please provide comma seprated values.')
    st.write('You might need to get rid of one of the values for the data for the output to appear.')

st.subheader("About The Data With Table:")
st.write(heart_data_table)
st.subheader("Model Performance on Training Data:")
st.write(training_data_accuracy)
st.subheader("Model Performance on Test Data:")
st.write(test_data_accuracy)
