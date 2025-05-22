import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title('Housing Price Prediction')
model=pickle.load(open('rf_model_class_2_nn.pkl','rb'))

# get the data from User
st.header('Housing Price Prediction given by User')
user_input={}

L=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup',
       'Latitude', 'Longitude']
for i in L:
    user_input[i]=st.number_input(i)
user_input_df=pd.DataFrame(user_input,index=[0])
prediction=model.predict(user_input_df)
st.subheader(f'Predicted Price is $ {prediction[0] * 100000}')