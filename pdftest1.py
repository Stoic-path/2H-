import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from openai import OpenAI
import PyPDF2
import xml.etree.ElementTree as ET

# Cargar variables de entorno
load_dotenv()
logger.info(os.getenv('GROQ_API_KEY'))

# Inicializar clientes de IA
qclient = Groq()
client = OpenAI()

# Configurar interfaz de usuario con Streamlit
st.title('PREDICCIÓN ELECTORAL')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for messages in st.session_state.messages:
    with st.chat_message(messages['role']):
        st.markdown(messages['content'])


def process_data(chat_completion) -> str:
    """ Procesa la respuesta del modelo en streaming """
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def extract_text_from_pdf(pdf_file):
    """ Extrae el texto de un archivo PDF. """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text


def extract_text_from_xml(xml_file):
    """ Extrae el texto de un archivo XML. """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    text = ""
    for elem in root.iter():
        if elem.text:
            text += elem.text.strip() + " "
    return text


def load_xlsx(file):
    """ Carga y procesa un archivo XLSX. """
    df = pd.read_excel(file)
    df.columns = df.columns.str.strip()
    st.write("Columnas detectadas:", df.columns)
    return df


def analyze_votes(df):
    """ Realiza el análisis de los votos en la data. """
    if 'Votos' not in df.columns:
        st.error("Error: La columna 'Votos' no existe en el archivo. Verifica los nombres de las columnas.")
        return None
    df['Votos'] = df['Votos'].astype(str).str.strip()
    df['Votos'] = df['Votos'].replace({'nan': 'Nulo', '': 'Nulo'})
    vote_counts = df['Votos'].value_counts(dropna=False)
    return vote_counts


def plot_votes(vote_counts):
    """ Grafica los resultados de los votos. """
    fig, ax = plt.subplots()
    vote_counts.plot(kind='bar', ax=ax)
    plt.title('Distribución de votos')
    plt.xlabel('Candidato')
    plt.ylabel('Cantidad de votos')
    st.pyplot(fig)

def split_text(text, max_length=8192, overlap=50): #Groq Llama 3-8B permite hasta 8192
    # tokens
    #Fragmentos pequeños (max_length < 1000)	50 - 100 caracteres
    #Fragmentos medianos (max_length ≈ 4000)	100 - 500 caracteres
    #Fragmentos grandes (max_length > 8000)	500 - 1000 caracteres
    """ Divide el texto en fragmentos con solapamiento para mejor coherencia. """
    sentences = text.split('. ')
    chunks = []
    chunk = ""

    for sentence in sentences:
        if len(chunk) + len(sentence) < max_length:
            chunk += sentence + ". "
        else:
            chunks.append(chunk.strip())
            chunk = sentence + ". "

    # Agregar solapamiento en cada fragmento excepto el primero
    if overlap > 0:
        for i in range(1, len(chunks)):
            chunks[i] = chunks[i - 1][-overlap:] + chunks[i]

    if chunk:
        chunks.append(chunk.strip())

    return chunks

# Carga de archivos
uploaded_files = st.file_uploader('Sube archivos PDF, XML o XLSX', type=['pdf', 'xml', 'xlsx'],
                                  accept_multiple_files=True)

if uploaded_files:
    extracted_text = ""
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        with st.chat_message('user'):
            st.markdown(f"Procesando: {uploaded_file.name}")

        if file_extension == 'pdf':
            extracted_text += extract_text_from_pdf(uploaded_file) + "\n\n"
        elif file_extension == 'xml':
            extracted_text += extract_text_from_xml(uploaded_file) + "\n\n"
        elif file_extension == 'xlsx':
            df = load_xlsx(uploaded_file)
            st.write("Vista previa de los datos:")
            st.write(df.head())
            vote_counts = analyze_votes(df)
            if vote_counts is not None:
                plot_votes(vote_counts)
                st.write("Resumen de votos:", vote_counts)

    st.session_state.messages.append(
        {'role': 'user', 'content': f'Archivos subidos: {", ".join([file.name for file in uploaded_files])}'}
    )

    chunks = split_text(extracted_text, max_length=500, overlap=50)  # Implementación de Overlap
    summaries = []

    for chunk in chunks:
        with st.chat_message('assistant'):
            stream_response = qclient.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Del archivo subido selecciona una muestra, luego clasifica en: Votos Noboa, Votos Luisa, Votos nulos. Luego presenta quién recibirá más votos. Luego cuenta los votos nulos y proporciona una conclusión.",
                    },
                    {
                        "role": "user",
                        "content": chunk,
                    },
                ],
                model="llama3-8b-8192",
                stream=True
            )

            response = process_data(stream_response)
            summary = st.write_stream(response)
            summaries.append(summary)

    final_summary = "\n\n".join(summaries)
    st.session_state.messages.append({'role': 'assistant', 'content': final_summary})

