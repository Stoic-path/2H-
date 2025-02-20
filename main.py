import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from loguru import logger
from openai import OpenAI

load_dotenv() #se cargan todas las variables que están en el .env

logger.info(os.getenv('GROQ_API_KEY'))#se llama a las variables de entorno con el os.getenv()

qclient = Groq() #instanciar Groq
client = OpenAI() #instanciar OpenAI

st.title('LLM CHAT')

#Generar chat
if 'messages' not in st.session_state: #setear los mensajes para que se simule un chat, sistema, usuario, asistente, usuario, asistente..
    st.session_state.messages = []

for messages in st.session_state.messages:
    with st.chat_message(messages['role']): # Se pone en el chat el rol
        st.markdown(messages['content']) # la respuesta, lo que se escribio o se responde


def process_data(chat_completion) -> str: #iterador parcial con un for para arreglar cuando muestra chunks código.Se utiliza para IA, se le va proporcionando
    #de 1 en 1. Xq lo que hace es que no se carge en memoria
    for chunk in chat_completion:
        if chunk.choices[0].delta.content: yield chunk.choices[0].delta.content #yield permite traer de token en token


if prompt := st.chat_input('Insert questions'): #Para ingresar el prompt manualmente por usuario
    with st.chat_message('user'): # se manda a st.chatmessage para ser mostrado.
        st.markdown(prompt)

    st.session_state.messages.append({'role': 'user', 'content': prompt}) #se agrega al historial mostrado en frontend

    with st.chat_message('assistant'):
        #stream_response = client.chat.completions.create(
        stream_response = qclient.chat.completions.create(
            messages=[
                {
                    "role": "system", #configuracion de como funciona el modelo, que hace el modelo
                    #"content": "Give the number in binary but only show me the number, not the process to obtain that number. ",
                    #"content": "Convert the given number to binary and return only the binary number as output. Do not provide any explanation, process, or additional text. Only return the final binary number.",
                    "content": "Answer the question",
                },
                {
                    "role": "user",
                    "content": prompt, #se proporciona el prompt
                },
            ],
            #model="deepseek-r1-distill-llama-70b", #cadena de pensamiento. Devuelve la meta data del pensamiento,proceso.
            #model="gpt-4o-mini",
            model="llama-3.3-70b-versatile",
            stream=True
        )

        response = process_data(stream_response)

        response = st.write_stream(response)

    st.session_state.messages.append({'role': 'assistant', 'content': response})

    #print(chat_completion.choices[0].message.content)

#en terminal: streamlit run main.py
#tensorflow #pytorch

