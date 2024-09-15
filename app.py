import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the API key
groq_api_key = os.getenv('GROQ_API_KEY')

# Title of the Web Page
st.set_page_config(
    page_title="StudySage | Wise answers for better grades"
)

# Initialize session state variables if not already initialized
if 'loader' not in st.session_state:
    st.session_state.loader = PyPDFLoader("Computer Education 11.pdf")

if 'docs' not in st.session_state:
    st.session_state.docs = st.session_state.loader.load_and_split()

# Set chunk size and overlap
chunk_size = 1000
chunk_overlap = 200

if 'text_splitter' not in st.session_state:
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        separators=["\n\n", ""],
        chunk_overlap=chunk_overlap
    )

if 'final_documents' not in st.session_state:
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

if 'embeddings' not in st.session_state:
    # Load HuggingFace embeddings only once
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5')

if 'vectors' not in st.session_state:
    # Create FAISS index for document retrieval, only once
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Set up the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Prepare the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    Provide a clear and detailed explanation of the topic in simple and easy-to-understand language.
    Answer the questions based on the provided context only and expand with relevant knowledge.
    Ensure your response is accurate and informative, addressing the question directly.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Get user input
user_question = st.text_input("Enter Your Question")

if user_question:
    # Create the document chain once
    document_chain = create_stuff_documents_chain(llm, prompt_template)

    # Use the previously created FAISS retriever (no need to recreate)
    retriever = st.session_state.vectors.as_retriever()

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Invoke the chain with the user's question
    response = retrieval_chain.invoke({'input': user_question})

    # Display the answer
    st.write(response['answer'])
