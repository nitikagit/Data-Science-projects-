import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler



MOCK_TRAIN_COLUMNS = ['height_cm', 'weight_kg', 'gender_female', 'gender_male',
       'category_traditional', 'category_western', 
       'skin_tone_black', 'skin_tone_brown', 'skin_tone_dusky',
       'skin_tone_white', 'body_shape_hourglass',
       'body_shape_inverted triangle', 'body_shape_rectangle',
       'body_shape_square', 'body_shape_triangle']


def user_input(gender, height_cm, weight_kg, skin_tone, body_shape, category):
    
    # 1. Create a temporary DataFrame from the simple user inputs
    new_user_data = pd.DataFrame({
        'gender': [gender], 
        'height_cm': [height_cm], 
        'weight_kg': [weight_kg], 
        'skin_tone': [skin_tone], 
        'body_shape': [body_shape],
        'category': [category]  
    })
    
    # 2. Apply One-Hot Encoding to the categorical features
    new_user_encoded = pd.get_dummies(new_user_data, drop_first=True)
    
    # 3. CRUCIAL STEP: Align columns with the full set of training features
    # This ensures the new input has all 27 columns in the correct order, 
    aligned_input = new_user_encoded.reindex(columns=MOCK_TRAIN_COLUMNS, fill_value=0)
    return aligned_input

##now model UI
    
with open("xyz.pkl","rb") as file:
    m=pickle.load(file)
with open("fitted_label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)
st.title("OUTFIT  RECOMMENDER  ")
st.write("SELECT YOUR GENDER HERE -->>")
g=st.selectbox("Choose Gender", ['male', 'female'])

st.write("SELECT YOUR CATEGORY HERE -->>")
c=st.selectbox("Choose category", ['traditional', 'western'])

st.write("SELECT YOUR HEIGHT  HERE -->>")
h=st.slider("Select height (ex.height is 5 feet then choose 0.5)",min_value=0.0,max_value=1.0,value=0.5,step=0.1)

st.write("SELECT YOUR WEIGHT  HERE -->>")
w=st.slider("Select weight (ex.weight is 50 kg  then choose 0.5)",min_value=0.0,max_value=1.0,value=0.5,step=0.1)

st.write("SELECT YOUR SKIN TONE HERE -->>")
skin=st.selectbox("Choose skin tone", ['brown', 'white','dusky','black'])

st.write("SELECT YOUR BODY SHAPE HERE -->>")
body=st.selectbox("Choose body shape", ['hourglass', 'triangle','inverted triangle','square','rectangle'])

user_input_df = user_input(
    gender=g, height_cm=h, weight_kg=w, skin_tone=skin, 
    body_shape=body, category= c
)
predi=st.button("RECOMMENDATION....")
prediction = m.predict(user_input_df)
predicted_outfit = label_encoder.inverse_transform([prediction])
if predi:
    if predicted_outfit=="saree":
        st.write(predicted_outfit)
        st.image("image1.png", caption="A Local Image", width=300)
        st.image("image 16.png", caption="A Local Image", width=300)
        st.image("image 22.png", caption="A Local Image", width=300)
       
    elif predicted_outfit=="kurta-pajamas":
        st.write(predicted_outfit)
        st.image("image 2.png", caption="A Local Image", width=300)
        st.image("image 13.png", caption="A Local Image", width=300)
        st.image("image 21.png", caption="A Local Image", width=300)
    elif predicted_outfit=="jeans-top":
        st.write(predicted_outfit)
        st.image("image 3.png", caption="A Local Image", width=300)
        st.image("image 18.png", caption="A Local Image", width=300)
        
    elif predicted_outfit=="shirts-pants":
        st.write(predicted_outfit)
        st.image("image 4.png", caption="A Local Image", width=300)
        st.image("image 11.png", caption="A Local Image", width=300)
    elif predicted_outfit=="lehnga":
        st.write(predicted_outfit)
        st.image("image 5.png", caption="A Local Image", width=300)
        st.image("image 24.png", caption="A Local Image", width=300)
    elif predicted_outfit=="coat-pants":
        st.write(predicted_outfit)
        st.image("image 6.png", caption="A Local Image", width=300)
        st.image("image 17.png", caption="A Local Image", width=300)
    elif predicted_outfit=="frocks":
        st.write(predicted_outfit)
        st.image("image 20.png", caption="A Local Image", width=300)
        st.image("image 23.png", caption="A Local Image", width=300)
        st.image("image 7.png", caption="A Local Image", width=300)
    elif predicted_outfit=="suits-pajamas":
        st.write(predicted_outfit)
        st.image("image 8.png", caption="A Local Image", width=300)
        st.image("image 19.png", caption="A Local Image", width=300)
    elif predicted_outfit=="gowns":
        st.write(predicted_outfit)
        st.image("image 9.png", caption="A Local Image", width=300)
        st.image("image 25.png", caption="A Local Image", width=300)
    elif predicted_outfit=="mini-skirt":
        st.write(predicted_outfit)
        st.image("image 10.png", caption="A Local Image", width=300)
    
    elif predicted_outfit=="long_frock":
        st.write(predicted_outfit)
        st.image("image 12.png", caption="A Local Image", width=300)
    elif predicted_outfit=="top-shorts":
        st.write(predicted_outfit)
        st.image("image 14.png", caption="A Local Image", width=300)
        st.image("image 26.png", caption="A Local Image", width=300)
    else:
        st.image("image 2.png", caption="A Local Image", width=300)
    


    
    
    


    
    