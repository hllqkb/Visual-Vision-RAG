import streamlit as st
import base64
from tools import ImageCaptionTool, ObjectDetectionTool
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from tempfile import NamedTemporaryFile
import os

# How to run the app:
# streamlit run main.py

##############################
### initialize agent #########
##############################

# initialize agent
conservation_memory = ConversationBufferMemory(
    memory_key="chat_history",
    k=10,
    return_messages=True
)
llm=ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key='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',
    temperature=0,
    model_name="deepseek/deepseek-chat-v3-0324:free"
)
agent = initialize_agent(
    agent="chat-conversational-react-description",
    tools=[ImageCaptionTool(), ObjectDetectionTool()],
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=conservation_memory,
    early_stopping_method='generate'
)






st.set_page_config(page_title="Chat with an Image", page_icon=":camera:", layout="wide")
st.header("Chat with an Image")
def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)
background_image_path = r"C:\Users\hllqkb\Desktop\Learn-Computer-Vision-in-30-Days\Learn-Computer-Vision-in-30-Days\pneumonia-classification-web-app\bg\bg1.jpg"
set_background(background_image_path)

file=st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if file:
    # display image
    st.image(file, use_column_width=True)

    # text input
    user_question = st.text_input('Ask a question about your image:')

    ##############################
    ### compute agent response ###
    ##############################
    with NamedTemporaryFile(dir='.', delete=False) as f:
        f.write(file.getbuffer())
        image_path = f.name

    # now we can run the agent on the image
    if user_question and user_question != "":
        with st.spinner(text="In progress..."):
            response = agent.run('{}, this is the image path: {}'.format(user_question, image_path))
            print(response)
            st.write(response)

    # we can manually delete the temporary file
    os.remove(image_path)
