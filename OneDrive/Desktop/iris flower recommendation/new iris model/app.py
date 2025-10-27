import pickle
import streamlit as st
import numpy as np
with open("xyz.pkl","rb") as file:
    m=pickle.load(file)
def iris_flower():
    st.title("Identification Of Iris")
    a=st.number_input("ENTER VALUE OF SAPLE LENGTH")
    b=st.number_input("ENTER VALUE OF SAPLE WIDTH")
    c=st.number_input("ENTER VALUE OF PETAL LENGTH")
    d=st.number_input("ENTER VALUE OF PETAL WIDTH")
    list=[a,b,c,d]
    array1=np.array(list)
    array_2d=np.array(array1).reshape(1,-1)
    predi=st.button("PREDICTION")
    val= m.predict(array_2d)
    if predi:
        if val[0]==0:
            st.write("SETOSA")
            st.image("setosa.jpeg", caption="Iris Setosa", use_column_width=True)
        elif val[0]==1:
            st.write("VERSICOLOR")
            st.image("versicolor.jpeg", caption="Iris Setosa", use_column_width=True)
        else:
            st.write("VERGINICA")
            st.image("verginica.jpeg", caption="Iris Setosa", use_column_width=True)
            
        
 
iris_flower()