import streamlit as st
import google.generativeai as genai
import os
import PIL.Image as Image
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
# st.write(api_key)
if (api_key is None):
    st.error("Please set the GOOGLE_API_KEY environment variable.")
    st.stop()


genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-1.5-flash-latest")

def analyze_human_image(image):
    prompt="""
    You are an AI trained to analyze human attributes from images with high accuracy. 
    Carefully analyze the given image and return the following structured details:

    You have to return all results as you have the image, don't want any apologize or empty results.

    - **Gender** (Male/Female/Non-binary)
    - **Age Estimate** (e.g., 25 years)
    - **Ethnicity** (e.g., Asian, Caucasian, African, etc.)
    - **Mood** (e.g., Happy, Sad, Neutral, Excited)
    - **Facial Expression** (e.g., Smiling, Frowning, Neutral, etc.)
    - **Glasses** (Yes/No)
    - **Beard** (Yes/No)
    - **Hair Color** (e.g., Black, Blonde, Brown)
    - **Eye Color** (e.g., Blue, Green, Brown)
    - **Headwear** (Yes/No, specify type if applicable)
    - **Emotions Detected** (e.g., Joyful, Focused, Angry, etc.)
    - **Confidence Level** (Accuracy of prediction in percentage)
    - **Beautyfull Level** (Accuracy of prediction in percentage)
    """

    response=model.generate_content([prompt, image]).text
    return response

st.set_page_config(
    page_title="Human Attribute Detection",
    page_icon="🕵🏼‍♂️"
)

st.title("Human Attribute Detection")
st.write("Upload an image and select the attributes you want to detect.")


image_upload=st.file_uploader("Upload a Imge", type=["jpg", "jpeg", "png"])

if (image_upload):
    image=Image.open(image_upload)
    image_info=analyze_human_image(image)

    col1,col2=st.columns(2)

    with col1:
        st.image(image=image, caption="Human Image",use_container_width=True)
    with col2:
        st.write(image_info)