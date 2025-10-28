import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# --- Estilos con degradados fríos ---
st.markdown("""
    <style>
        /* Fondo general */
        .stApp {
            background: linear-gradient(135deg, #e3f2fd 0%, #e8eaf6 50%, #f3e5f5 100%);
            font-family: 'Poppins', sans-serif;
        }

        /* Título principal */
        h1 {
            background: linear-gradient(90deg, #4f6da3, #7c8dc3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            font-weight: 700;
            font-size: 2.3em;
            margin-bottom: 0.2em;
        }

        /* Subtítulos */
        h2, h3 {
            color: #607d8b;
            font-weight: 600;
        }

        /* Barra lateral */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e8eaf6 0%, #e3f2fd 100%);
            border-radius: 15px;
            color: #455a64;
        }

        /* Cuadros de texto */
        .stTextInput > div > div > input, textarea {
            border-radius: 10px !important;
            border: 1px solid #b0bec5 !important;
            background-color: #ffffff !important;
        }

        /* Botones */
        button[kind="primary"] {
            background: linear-gradient(90deg, #90caf9, #64b5f6);
            color: white !important;
            border-radius: 10px !important;
            border: none !important;
            font-weight: 600 !important;
            box-shadow: 0px 4px 10px rgba(100, 149, 237, 0.3);
        }

        button[kind="primary"]:hover {
            background: linear-gradient(90deg, #7eb8eb, #58a6e0);
        }

        /* Imagen centrada */
        [data-testid="stImage"] img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 15px;
            box-shadow: 0px 0px 15px rgba(90, 130, 180, 0.4);
        }

        /* Cuadros de información */
        .stAlert {
            border-radius: 12px !important;
            background: linear-gradient(180deg, #e3f2fd 0%, #f3e5f5 100%);
        }

        /* Texto y markdown */
        .stMarkdown p {
            color: #37474f;
            font-size: 1.05em;
        }
    </style>
""", unsafe_allow_html=True)

# --- Título y descripción ---
st.title('Generación Aumentada por Recuperación (RAG)')
st.write(f"Versión de Python: **{platform.python_version()}**")

# --- Imagen decorativa (cambia el nombre a 'cinna.jpg') ---
try:
    image = Image.open('cinna.jpg')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# --- Barra lateral informativa ---
with st.sidebar:
    st.subheader("Asistente de Análisis de PDF")
    st.markdown("""
    Este agente inteligente te ayudará a:
    - Analizar el contenido de un documento PDF  
    - Responder preguntas basadas en el texto  
    - Resumir y explicar fragmentos específicos  

    *Solo necesitas subir tu archivo y escribir tu pregunta.*
    """)

# --- Clave de API ---
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# --- Cargar PDF ---
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# --- Procesamiento del PDF ---
if pdf is not None and ke:
    try:
        # Extraer texto
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        # Dividir texto
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")

        # Crear embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Interfaz de pregunta
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        # Procesar pregunta
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            # Mostrar respuesta
            st.markdown("### Respuesta del asistente:")
            st.markdown(f"<div style='background: linear-gradient(180deg, #e8eaf6 0%, #e3f2fd 100%); padding:15px; border-radius:10px; color:#37474f;'>{response}</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar.")
else:
    st.info("Carga un archivo PDF para comenzar tu análisis.")
