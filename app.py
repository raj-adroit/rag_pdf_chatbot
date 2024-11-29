import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

class PDFChatBot:
    def __init__(self, openai_api_key):
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
         # Add persistent path for Chroma
        self.persist_directory = "chroma_db"
        
    def load_pdfs(self, uploaded_files):
        documents = []
        for uploaded_file in uploaded_files:
            # Save uploaded file temporarily
            with open(f"temp_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(f"temp_{uploaded_file.name}")
            documents.extend(loader.load())
            # Clean up temporary file
            os.remove(f"temp_{uploaded_file.name}")
        
        # Split documents into chunks
        splits = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Create conversation chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory
        )
        # self.vectorstore.persist()
        
    def ask_question(self, question):
        if not hasattr(self, 'qa_chain'):
            return "Please upload PDFs first."
        
        response = self.qa_chain.invoke({"question": question})
        return response['answer']

# Streamlit UI
st.set_page_config(page_title="PDF ChatBot", page_icon="📚")
st.title("PDF ChatBot")

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = PDFChatBot(os.getenv("OPEN_API_KEY"))
if 'messages' not in st.session_state:
    st.session_state.messages = []

# File uploader
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

if uploaded_files:
    if st.button("Process PDFs"):
        with st.spinner("Processing PDFs..."):
            st.session_state.chatbot.load_pdfs(uploaded_files)
        st.success("PDFs processed successfully!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your PDFs"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        response = st.session_state.chatbot.ask_question(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
