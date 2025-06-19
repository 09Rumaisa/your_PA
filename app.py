import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langsmith import Client
from langchain.schema.runnable import RunnableMap
from dotenv import load_dotenv
import os
import tempfile

# --- Load API keys ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
smithkey = os.getenv("langsmithkey")

client = Client(api_key=smithkey)

# --- LangSmith Tracing ---
if not smithkey:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    print("⚠ LangSmith key not found. Tracing disabled.")
else:
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = smithkey.strip()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "your_PA"
    print("✅ LangSmith tracing enabled")

# --- Set up LLM ---
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

# --- Chat History Store ---
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Streamlit Config ---
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("🤖 Chat with your PDF")
st.sidebar.header("Upload your PDF")

# --- Upload PDF ---
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# --- Streamlit session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Process uploaded PDF ---
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    texts = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(texts, embeddings)
    retriever = vector_store.as_retriever()

    # --- Pull RAG prompt from LangSmith hub ---
    prompt_runnable = client.pull_prompt("rlm/rag-prompt", include_model=True)

    # --- Create RAG Chain ---
    rag_chain = RunnableMap({
        "question": lambda x: x["question"],
        "context": lambda x: "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(x["question"])])
    }) | prompt_runnable | llm

    # --- Wrap with Chat History support ---
    qa_with_history = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    # --- Display chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # --- Chat input box at bottom ---
    prompt = st.chat_input("Ask a question about the PDF...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use the RAG + LangSmith + History agent
                response = qa_with_history.invoke(
                    {"question": prompt},
                    config={"configurable": {"session_id": "default"}}
                )
                answer = getattr(response, "content", str(response))
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Upload a PDF from the sidebar to begin chatting.")
