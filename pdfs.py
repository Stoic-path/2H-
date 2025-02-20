import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from openai import OpenAI
import PyPDF2
import xml.etree.ElementTree as ET

load_dotenv()
logger.info(os.getenv('GROQ_API_KEY'))

qclient = Groq()
client = OpenAI()

st.title('LLM CHAT - Multi-file Summary')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for messages in st.session_state.messages:
    with st.chat_message(messages['role']):
        st.markdown(messages['content'])

def process_data(chat_completion) -> str:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def extract_text_from_pdf(pdf_file):
    """Extrae el texto de un archivo PDF."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_xml(xml_file):
    """Extrae el texto de un archivo XML."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    text = ""

    for elem in root.iter():
        if elem.text:
            text += elem.text.strip() + " "

    return text

uploaded_files = st.file_uploader('Upload PDF/XML files', type=['pdf', 'xml'], accept_multiple_files=True)

if uploaded_files:
    extracted_text = ""

    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        with st.chat_message('user'):
            st.markdown(f"Processing: {uploaded_file.name}")

        if file_extension == 'pdf':
            extracted_text += extract_text_from_pdf(uploaded_file) + "\n\n"
        elif file_extension == 'xml':
            extracted_text += extract_text_from_xml(uploaded_file) + "\n\n"

    st.session_state.messages.append({'role': 'user', 'content': f'Uploaded files: {", ".join([file.name for file in uploaded_files])}'})

    with st.chat_message('assistant'):
        stream_response = qclient.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You will only answer in Spanish. I will upload multiple files (PDFs and XMLs), and you will provide me with a summary.",
                },
                {
                    "role": "user",
                    "content": extracted_text,
                },
            ],
            model="llama3-8b-8192",
            stream=True
        )

        response = process_data(stream_response)
        response = st.write_stream(response)

    st.session_state.messages.append({'role': 'assistant', 'content': response})