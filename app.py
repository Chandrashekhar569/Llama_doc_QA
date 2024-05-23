import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

# Improved UI starts with loading the environment variables efficiently
load_dotenv()

# Load the groq and google api key
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Set page configuration for a wider layout and a title
st.set_page_config(page_title="Llama Model Q&A")

# Use a more descriptive title with a larger font size
st.title("📚 Llama Model Document Q&A")

# Sidebar with personal links and professional profile
st.sidebar.title("About Me")
st.sidebar.markdown("Developed by **Chandrashekhar Chaudhari**")
st.sidebar.markdown("""
I'm Chandrashekhar Chaudhari, currently working as a Data Analyst.
I have expertise in:
- Python
- MySQL
- Power BI
- Machine Learning
- Deep Learning
- Generative AI
""")
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue)](https://www.linkedin.com/in/chandrashekhar1997/)")
st.sidebar.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Chandrashekhar569)")


# Initialize the LLM with the API key
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the prompt template with a clear structure
prompt = ChatPromptTemplate.from_template("""
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Function to create vector embeddings
def vectore_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader("./Book")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Input for user's question with a clearer prompt
prompt1 = st.text_input("🔍 Ask a question about the book content:")

# Button to create vector store with a more engaging label
if st.button("🚀 Build Vector Store"):
    vectore_embedding()
    st.success("Vector Store database is ready!")

# Display the response when a question is asked
if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(response['answer'])

    # Use an expander for document similarity search to reduce clutter
    with st.expander("📄 Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("---------------------------------")
