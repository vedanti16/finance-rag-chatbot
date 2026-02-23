import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def create_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key="sk-xxxxx")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local("faiss_index")

if __name__ == "__main__":
    create_vector_store("sample_financial_report.pdf")
