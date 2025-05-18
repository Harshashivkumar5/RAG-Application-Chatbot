import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# ====== API Key Setup ======
# Replace with your key or set it via Streamlit secrets
# e.g., st.secrets["OPENAI_API_KEY"]
os.environ['OPENAI_API_KEY'] = st.secrets.get("OPENAI_API_KEY", "your-openai-key")

# ====== PDF Ingestion ======
PDF_PATH = "scientific.pdf"  # Ensure this file is in the same folder

@st.cache_resource
def load_qa_chain(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    retriever = db.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    llm = ChatOpenAI(temperature=0)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return chain

qa_chain = load_qa_chain(PDF_PATH)

# ====== Streamlit UI ======
st.title("🧠 PDF Chatbot with Memory")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask something about the PDF:")

if query:
    with st.spinner("Thinking..."):
        result = qa_chain({
            "question": query,
            "chat_history": st.session_state.chat_history
        })
        st.session_state.chat_history.append((query, result['answer']))
        st.write(f"**You:** {query}")
        st.write(f"**Bot:** {result['answer']}")

# Show full history
if st.session_state.chat_history:
    with st.expander("🔁 Chat History"):
        for i, (q, a) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You {i+1}:** {q}")
            st.markdown(f"**Bot {i+1}:** {a}")
