import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
import google.auth
from google.auth.transport.requests import Request
import os

# Set up Google Cloud credentials
def setup_credentials():
    """
    Sets up Google Cloud credentials.  This will attempt to use
    application default credentials, but if those aren't found, it
    will try to use the credentials from the Streamlit secrets.
    """
    try:
        # Attempt to get default credentials.  This will work if you've
        # already authenticated with gcloud.
        credentials, _ = google.auth.default()
        if credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        return credentials
    except google.auth.exceptions.DefaultCredentialsError:
        # If default credentials aren't found, try to use the credentials
        # from Streamlit secrets.
        if "GOOGLE_CREDENTIALS" in st.secrets:
            # Write the credentials to a temporary file.
            creds = st.secrets["GOOGLE_CREDENTIALS"]
            import json
            creds_json = json.dumps(creds)
            # Use a filename that is unlikely to collide
            temp_file_name = "temp_google_credentials.json"
            with open(temp_file_name, "w") as f:
                f.write(creds_json)
            #tell the os where the credentials file is.
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_name
            credentials, _ = google.auth.default()
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            return credentials
        else:
            st.error("No Google Cloud credentials found.  Please set up Application Default Credentials or add your credentials to Streamlit secrets as 'GOOGLE_CREDENTIALS'.")
            return None

# Initialize Vertex AI
def initialize_vertex_ai(project_id, location):
    """Initializes Vertex AI.
    Args:
        project_id: The Google Cloud project ID.
        location: The Google Cloud region.
    """
    import vertexai
    try:
        vertexai.init(project=project_id, location=location)
    except Exception as e:
        st.error(f"Error initializing Vertex AI: {e}")
        return False
    return True

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    """
    Extracts the text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text, or an empty string on error.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:  #open in binary mode
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""  # handle None
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""
    return text

# Load data and create chain
@st.cache_resource
def load_chain(pdf_path, project_id, location):
    """
    Loads the data from the PDF, splits it into chunks, creates embeddings,
    stores them in a vector store, and creates a conversational retrieval chain.

    Args:
        pdf_path (str): Path to the PDF file.
        project_id (str): Google Cloud Project ID.
        location (str): Google Cloud Location.

    Returns:
        ConversationalRetrievalChain: The created chain.  Returns None on error.
    """
    # 1. Extract text from the PDF
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        return None  # Error already shown by extract_text_from_pdf

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
    )
    documents = text_splitter.create_documents([raw_text])

    # 3. Create embeddings using GoogleGenerativeAIEmbeddings
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        return None

    # 4. Create vector store (FAISS)
    try:
        vectorstore = FAISS.from_documents(documents, embeddings)
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None

    # 5. Create the LLM using ChatGoogleGenerativeAI
    try:
        llm = ChatGoogleGenerativeAI(model_name="models/chat-bison-002", temperature=0.5) #temperature 0.5 is better
    except Exception as e:
        st.error(f"Error creating LLM: {e}")
        return None

    # 6. Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    # 7. Create the Conversational Retrieval Chain
    try:
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory,
            return_source_documents=True, # Include source documents in the response
        )
    except Exception as e:
        st.error(f"Error creating ConversationalRetrievalChain: {e}")
        return None
    return chain

# Main function
def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Gemini PDF Chatbot")

    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        pdf_file = st.file_uploader("Upload your PDF", type="pdf")
        project_id = st.text_input("Google Cloud Project ID", st.session_state.get("project_id", ""))
        location = st.text_input("Google Cloud Location", st.session_state.get("location", "us-central1"))

        # Store project_id and location in session state
        st.session_state.project_id = project_id
        st.session_state.location = location
        creds = setup_credentials() #setup the credentials

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Process PDF and initialize chain
    if pdf_file is not None:
        # Save the uploaded file to a temporary location
        temp_pdf_path = "temp.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_file.read())

        if not initialize_vertex_ai(project_id, location):
            st.stop()  # Stop if Vertex AI initialization fails

        chain = load_chain(temp_pdf_path, project_id, location)
        if chain is None:
            st.stop() #stop if chain is not loaded.

        #delete the temp file.
        os.remove(temp_pdf_path)

        # Chat input and query
        query = st.text_input("Ask a question about the PDF:")
        if query:
            with st.spinner("Processing your request..."):
                try:
                    result = chain({"question": query, "chat_history": st.session_state.chat_history})
                    answer = result["answer"]
                    source_documents = result.get("source_documents", [])  # Get source documents

                    st.session_state.chat_history.append((query, answer))
                    st.write(f"**You:** {query}")
                    st.write(f"**Gemini:** {answer}")

                    # Display source documents
                    if source_documents:
                        st.write("**Source Documents:**")
                        for doc in source_documents:
                            st.write(f"- Page {doc.page_content}") #changed from page number to page content

                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.info("Please upload a PDF file to get started.")

    # Display chat history
    if st.session_state.chat_history:
        with st.expander("Chat History"):
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**You {i+1}:** {q}")
                st.markdown(f"**Gemini {i+1}:** {a}")

if __name__ == "__main__":
    main()

