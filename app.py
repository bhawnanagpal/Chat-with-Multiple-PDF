import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Validate API key
if not api_key:
    st.error("API Key not found. Please set your GOOGLE_API_KEY in the environment variables.")
    st.stop()

genai.configure(api_key=api_key)

def extract_text_from_pdfs(pdf_files):
    """Extracts and returns text from uploaded PDFs."""
    text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def split_text_into_chunks(text, chunk_size=10000, chunk_overlap=1000):
    """Splits text into manageable chunks for processing."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def create_vector_store(text_chunks):
    """Creates and saves FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    """Creates a conversational chain with a custom prompt."""
    prompt = PromptTemplate(
        template="""
        Answer the question as accurately as possible from the provided context.
        If the answer is not available, say "Answer is not available in the context."
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """,
        input_variables=["context", "question"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def summarize_text(text):
    """Summarizes given text using Gemini AI model."""
    summarizer = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(
        template="""
        Provide a detailed and comprehensive summary of the following text:
        
        Text: {context}
        
        Detailed Summary:
        """,
        input_variables=["context"]
    )
    chain = load_qa_chain(summarizer, chain_type="stuff", prompt=prompt)
    summary = chain({"input_documents": [Document(page_content=text)]}, return_only_outputs=True)
    return summary.get("output_text", "No summary generated.")

def answer_user_question(user_question):
    """Processes user input and generates a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response.get("output_text", "No response generated.")

def main():
    st.set_page_config(page_title="Chat with PDF")
    st.header("Chat with PDF using Gemini 💁")
    
    with st.sidebar:
        st.title("Menu")
        pdf_files = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_files:
                with st.spinner("Processing PDFs..."):
                    text = extract_text_from_pdfs(pdf_files)
                    text_chunks = split_text_into_chunks(text)
                    create_vector_store(text_chunks)
                    st.success("Processing complete! Ask your questions below.")
            else:
                st.warning("Please upload at least one PDF.")
    
    # Summarization section
    pdf_names = [pdf.name for pdf in pdf_files] if pdf_files else []
    pdf_input = st.text_input("Enter PDF names for summarization (comma-separated)")
    if st.button("Summarize"):
        selected_pdfs = [name.strip() for name in pdf_input.split(",")]
        combined_text = "".join(extract_text_from_pdfs([pdf for pdf in pdf_files if pdf.name in selected_pdfs]))
        if combined_text:
            summary = summarize_text(combined_text)
            st.write("Summary:", summary)
        else:
            st.error("Error: Selected PDFs not found or empty.")
    
    # Question-answering section
    user_question = st.text_input("Ask a question from the PDFs")
    if user_question:
        with st.spinner("Generating response..."):
            response = answer_user_question(user_question)
            st.write("Reply:", response)

if __name__ == "__main__":
    main()
