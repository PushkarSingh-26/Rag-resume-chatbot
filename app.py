import streamlit as st
import tempfile
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama  # Ollama LLM

# --------------------------
# Streamlit App UI
# --------------------------
st.set_page_config(page_title="Resume Q&A Bot", page_icon="🤖")
st.title("📄 Resume Q&A Chatbot (RAG with Ollama + HuggingFace)")

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------
# Upload Resume
# --------------------------
uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract text depending on format
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PdfReader(tmp_path)
        for page in pdf_reader.pages:
            text += page.extract_text()
    elif uploaded_file.name.endswith(".txt"):
        with open(tmp_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif uploaded_file.name.endswith(".docx"):
        from docx import Document
        doc = Document(tmp_path)
        text = "\n".join([p.text for p in doc.paragraphs])

    # --------------------------
    # Chunking
    # --------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # --------------------------
    # Embeddings + Vectorstore
    # --------------------------
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # --------------------------
    # Ollama LLM (Gemma 3B)
    # --------------------------
    llm = ChatOllama(model="gemma3:1b")  # ensure Ollama is running locally

    st.session_state.conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )
    st.success("✅ Resume uploaded and processed!")

# --------------------------
# Chat Interface
# --------------------------
if st.session_state.conversation:
    user_question = st.text_input("Ask a question about your resume:")

    if user_question:
        response = st.session_state.conversation(
            {"question": user_question, "chat_history": st.session_state.chat_history}
        )
        st.session_state.chat_history.append((user_question, response["answer"]))

    # Display Chat History
    for q, a in st.session_state.chat_history:
        st.write(f"**You:** {q}")
        st.write(f"**Bot:** {a}")
