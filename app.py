#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      mahal
#
# Created:     29/10/2024
# Copyright:   (c) mahal 2024
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import streamlit as st
import numpy as np
from PIL import Image
import pickle
import streamlit as st
import numpy as np
from PIL import Image
import pickle
pickle_in=open("model.pkl","rb")
model= pickle.load(pickle_in)
def predict(stories,bathrooms,parking):
    array=np.array(list,np.float64)
    prediction=model.predict([array])
    return prediction
def main():
    st.title("HOUSE_PRICE-PREDICTION")
    template="""
    <div style="background-color:black;padding:10px;font-size=23px;">
    <h1>style="color:white;text-align;center;>HOUSE PRICE-PREDICTION<h1>
    </div>
    """
    st.markdown(template,unsafe_allow_html=True)
    img=Image.open("house.jpg")
    st.image(img,width=300,caption="House-worth")
    house_age=st.text_input("STORIES","enter your house stories")
    distance_to_nearest_metro=st.text_input("NUMBER OF BATHROOMS","enter your number of bathrooms")
    no_of_nearby_stores = st.text_input("NUMBER OF PARKINGS","enter no. of PARKINGS")
    result=""
    if st.button("predict"):
        result=(predict(stories,bathrooms,parking))
        st.succes("the house predicted values is{}".format(result))
        if __name__=="__main__":
            main()