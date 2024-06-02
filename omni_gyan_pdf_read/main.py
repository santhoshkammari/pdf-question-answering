import json
import time
from typing import List

from langchain import hub
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_pdf(file_path: str, mode: str):
    pdf_loader = UnstructuredPDFLoader(file_path=file_path,
                                       mode=mode)
    return pdf_loader.load()


def load_and_save():
    __data: List[Document] = read_pdf(file_path="data/Omnigyan Library.pdf",
                                      mode="paged")
    json_data = []
    for __d in __data:
        _d: dict = {}
        _d["page_content"] = __d.page_content
        _d["type"] = __d.type
        _d["metadata"] = __d.metadata
        json_data.append(_d)
    with open("pdf_data.json", "w") as f:
        json.dump(json_data, f, indent=2)


def load_data_from_pdf_json():
    with open("pdf_data.json", "r") as f:
        pdf_data = json.load(f)
    __data = []
    for x in pdf_data:
        __data.append(Document(
            page_content=x["page_content"],
            metadata=x["metadata"],
            type=x["type"]
        ))
    return __data


def perform_splits(__data: List[Document]):
    __text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    __splits: List[Document] = __text_splitter.split_documents(documents=__data)
    return __splits




def load_vectorstore(load_from_disk: bool = False, splits=None):
    if load_from_disk:
        __vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings=OllamaEmbeddings(model=EMBED_MODEL),
                                         allow_dangerous_deserialization=True)
    else:
        print("start vector storing ..")
        __vectorstore = FAISS.from_documents(documents=splits,
                                             embedding=OllamaEmbeddings(model=EMBED_MODEL))
        __vectorstore.save_local(VECTOR_STORE_PATH)
        print("End vector storing ...")
    return __vectorstore


def format_docs(docs: List):
    return "\n\n".join(docs)


def llm_input(text):
    print('==========')
    print(text)
    print('==========')
    return text

CHAT_MODEL= "tinyllama"
EMBED_MODEL = 'nomic-embed-text'
VECTOR_STORE_PATH: str = "vectorstore_pdf"
CHUNK_SIZE = 100
CHUNK_OVERLAP = 50

def retriever(user_input,k):
    tokens_input = user_input.split()
    chunk_window = 4
    tokens = [" ".join(tokens_input[i:i + chunk_window]) for i in range(0, len(tokens_input)) if
              i + chunk_window <= len(tokens_input)]
    searches = []
    for t in tokens:
        searches.extend(vectorstore.similarity_search(t,k=k))
    # searches = [vectorstore.similarity_search(__chunk) for ]
    outputs = [_.page_content for _ in searches]  ## pagecontent list
    unique_outputs = list(set(outputs))
    __text = "\n\n".join(unique_outputs)
    return __text

if __name__ == '__main__':
    data: List[Document] = load_data_from_pdf_json()
    splits: List[Document] = perform_splits(__data=data)
    vectorstore = load_vectorstore(splits=splits, load_from_disk=True)
    # retriever = vectorstore.as_retriever()
    template = """
    You are the Document extraction Expert . Your task is to extract the information from the provided context.
    
    Context: {context}
    question: {user_input}
    
    to provide the answer, follow these steps:
    1. Understand the question to perform the extraction
    2. Identify the parts of the information required
    3. Extract the required information from the identified parts
    4. Respond with Answer.

    """
    prompt = PromptTemplate.from_template(
        template=template
    )
    llm = ChatOllama(model=CHAT_MODEL,max_tokens = 10)

    user_input = "start"
    while user_input!= "exit":
        user_input = input("Enter Text: \n")

        __chain_input = {"context":retriever(user_input,k=10),
                         "user_input":user_input}
        chain = prompt | llm_input | llm

        s = time.perf_counter()
        for _ in chain.stream(__chain_input):
            print(_.content,end="",flush=True)
        print(time.perf_counter()-s)