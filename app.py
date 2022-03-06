from heart_detection import *
import streamlit as st
#import matplotlib.pyplot as plt
from matplotlib.figure import Figure
#import cv2 as cv
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import cv2



hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.title("EpochZero AI Heart Detection")
#st.header("Brain Tumor MRI Classification")
st.text("Upload a chest x-ray Image to Detect the Heart")

uploaded_file = st.file_uploader("Upload a chest x-ray ...", type=['jpg','jpeg','png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded x-ray.', use_column_width=True)
    st.write("")
    st.write("Analyzing image...")

    img = image  # Select a subject

    img_channel_transforms = transforms.Compose([
                                transforms.Grayscale(num_output_channels=1),]) 

    img_transforms = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(0.49, 0.248),])

    img_1c = img_channel_transforms(img)   # transform to 1 channel image

    #print(img)

    img_np = np.asarray(img_1c) / 255

    img_array = cv2.resize(img_np, (224, 224)).astype(np.float32)


    processed_image = img_transforms(img_array) # transform to tensor and standardize


    data = processed_image

    heart = classify(data)    

    
    fig = Figure(figsize = (50, 50),
                 dpi = 100)
    
    

    plot1 = fig.add_subplot(121)
    plot2 = fig.add_subplot(122)
    plot1.imshow(processed_image[0], cmap="binary")
    #plot1.text(3, 8, f"{final_pred:.2f}%", style='italic', size = 50, bbox={'facecolor':'red' if final_pred > 50 else 'green', 'alpha':0.5, 'pad':10})


    #plot1.text(0.5, 0.5, 'EpochZero', transform=plot1.transAxes, fontsize=100, color='white', alpha=0.3, ha='center', va='center', rotation='0')

    
    plot2.imshow(processed_image[0], cmap="binary")

    
    plot2.add_patch(heart)


    #plot2.text(0.5, 0.5, 'EpochZero', transform=plot2.transAxes, fontsize=100, color='white', alpha=0.3, ha='center', va='center', rotation='0')
    
    
    st.pyplot(fig)

    

for i in range(18):
    st.write("")
    i += 1



st.write("By Mayowa Abejirin for EpochZero")


