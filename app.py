# import streamlit as st
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# import os
# import tempfile

# # Load API key
# load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")
# print("API Key:", openai_api_key)
# llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

# # Streamlit UI
# st.set_page_config(page_title="PDF QA App", layout="wide")
# st.title("📄 Ask Questions from a PDF")
# st.sidebar.header("Upload your document")

# # Sidebar: File uploader
# uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# # Check if file is uploaded
# if uploaded_file:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_file.read())
#         tmp_path = tmp_file.name

#     # Load and process PDF
#     loader = PyPDFLoader(tmp_path)
#     docs = loader.load()

#     # Text splitting
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
#     texts = text_splitter.split_documents(docs)

#     # Embeddings & Vector store
#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     vector_store = FAISS.from_documents(texts, embeddings)
#     retriever = vector_store.as_retriever()
#     qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

#     # Form input
#     with st.form("query_form"):
#         text = st.text_area("Ask a question based on the PDF:",
#                             "What is the CGPA of Rumaisa?")
#         submitted = st.form_submit_button("Submit")

#         if not openai_api_key or not openai_api_key.startswith("sk-"):
#             st.warning("Please set a valid OpenAI API key in your .env file!", icon="⚠")

#         if submitted and openai_api_key and openai_api_key.startswith("sk-"):
#             with st.spinner("Thinking..."):
#                 response = qa_chain.run(text)
#             st.success(response)


import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import tempfile

# Load API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

# Streamlit page config
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("🤖 Chat with your PDF")
st.sidebar.header("Upload your PDF")

# Sidebar: PDF upload
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load and process PDF
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (at bottom)
    prompt = st.chat_input("Ask a question about the PDF...")

    if prompt:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.run(prompt)
                st.markdown(response)

        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.info("👈 Upload a PDF from the sidebar to begin chatting.")
