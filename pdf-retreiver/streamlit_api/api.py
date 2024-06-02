import streamlit as st
import json
import time
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load the necessary functions from the original code
# (paste the function definitions here)

# Streamlit app
def main():
    st.title("PDF Question Answering")

    # Sidebar options
    st.sidebar.title("Options")
    load_data = st.sidebar.button("Load Data")
    if load_data:
        data = load_data_from_pdf_json()
        splits = perform_splits(data)
        vectorstore = load_vectorstore(splits=splits)

    user_input = st.text_input("Enter your question")
    if user_input:
        with st.spinner("Processing..."):
            context = retriever(user_input, k=10)
            chain_input = {"context": context, "user_input": user_input}
            template = """
            You are the question answer expert, your task is to provide an answer for the question from the options using the context.

            Context: {context}

            The question is in the form: question: option1, option2, option3, option4

            Examples:
            question: newgen services are: good, bad, nice, waste
            answer: good
            question: newgen is a --- company: big, well known, waster, normal
            answer: well known

            Now respond to the following:
            question: {user_input}
            answer:
            """
            prompt = PromptTemplate.from_template(template=template)
            llm = ChatOllama(model=CHAT_MODEL, max_tokens=10)
            chain = prompt | llm_input | llm

            st.subheader("Answer")
            s = time.perf_counter()
            for _ in chain.stream(chain_input):
                st.write(_.content, end="", flush=True)
            st.write(f"\nTime taken: {time.perf_counter() - s:.2f} seconds")

if __name__ == "__main__":
    main()