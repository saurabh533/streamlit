try:
    import streamlit as st
except ModuleNotFoundError:
    raise ImportError("The 'streamlit' module is not installed. Install it using 'pip install streamlit'.")

from langchain.vectorstores import Cassandra
from langchain.schema.vectorstore import VectorStoreRetriever
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain.embeddings import HuggingFaceEmbeddings
import cassio

# AstraDB Credentials
ASTRA_DB_KEYSPACE = "default_keyspace"  # Replace with your keyspace
ASTRA_DB_TABLE = "budget_embeddings"  # Replace with your table name
ASTRA_DB_ID = "7564b6e8-1f46-4f13-bfbc-6696c8a613bd"
API_TOKEN = "AstraCS:AZuHUEjcCinXgZmuNtierZFI:2f9d7193dffe1a610900ed786f9e089d4ed9590ae8ed36c4d2889933f4960cea"

# Initialize AstraDB connection
cassio.init(token=API_TOKEN, database_id=ASTRA_DB_ID)

# Define embedding function
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def retrieve_documents(query):
    """Retrieve relevant documents from AstraDB using LangChain's VectorStoreRetriever."""
    astra_vector_store = Cassandra(
        embedding=embedding_function,
        table_name=ASTRA_DB_TABLE,
        keyspace=ASTRA_DB_KEYSPACE,
    )
    retriever = astra_vector_store.as_retriever()
    results = retriever.get_relevant_documents(query)
    return results if results else "No relevant documents found."

def generate_response(query):
    """Generates a response using a locally running Ollama LLM."""
    context = retrieve_documents(query)
    if isinstance(context, str):
        return context  # Return early if no relevant documents found
    
    template = """Use the following context to answer the question: {context} Question: {query} Answer: """
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model="deepseek-r1:1.5b")
    chain = prompt | model  
    llm_output = chain.invoke({"context": context, "query": query})
    
    # Clean output text
    cleaned_txt = re.sub(r"<think>.*?</think>\s*", "", llm_output, flags=re.DOTALL)
    return cleaned_txt.strip()

# Streamlit UI
st.title("AstraDB-Powered AI Chatbot")
query = st.text_input("Enter your question:")
if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        response = generate_response(query)
        st.subheader("Response:")
        st.write(response)

