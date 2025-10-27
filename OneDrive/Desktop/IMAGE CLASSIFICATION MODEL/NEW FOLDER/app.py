import pickle
import streamlit as st
from PIL import Image
import numpy as np


with open("xyz.pkl","rb") as file:
    m=pickle.load(file)
def create_2d_array_from_image(upload_file, target_size=(512, 512)):
        # 1. Open the image
        img = Image.open(upload_file)
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized)
        flattened_array = img_array.flatten()
        reshaping=flattened_array.reshape(1,-1)
        return reshaping
def image_prediction():
    st.title("Image Classifier Model ")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
        st.write("Image uploaded successfully!")
        final_2d_array = create_2d_array_from_image(uploaded_file,(512,512))
        predi=st.button("PREDICTION STARTS....")
        im = m.predict(final_2d_array)
        if predi:
            if im[0]==0:
                st.write("DIVYA")
            elif im[0]==1:
                st.write("NITIKA")
        
            elif im[0]==2:
                st.write("JITENDER")
            elif im[0]==3:
                st.write("KESHAV")
            elif im[0]==4:
                st.write("MAHI")
            elif im[0]==5:
                st.write("LIBARARY MAM")
            else:
                print("OTHERS")


        
    else:
        # This message is shown when the app first loads or no file is selected
        st.info("Please upload an image file to see the result.")


if __name__ == "__main__":
    image_prediction()
