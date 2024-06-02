#  PDF Question Answering with LangChain and ChatOllama 🔥🔥

## **RESPONSE TIME : 12sec 🕛**

add fire sysnmbiad or make it like hurra y in guthb
## Features

- Load and preprocess PDF documents
- Create a vector store using FAISS and OllamaEmbeddings
- Retrieve relevant context from the vector store based on user input
- Generate answers using the ChatOllama model and a prompt template
- Interactive command-line interface for asking questions
- Streamlit app for a user-friendly web interface

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/pdf-question-answering.git
cd pdf-question-answering
```
2. Create a virtual environment and install the required dependencies:

```bash
python -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
pip install -r requirements.txt
```
## Usage
### Command-Line Interface

1. Place your PDF file(s) in the data directory.
2. Run the load_and_save function to preprocess the PDF and save the data to pdf_data.json:

```python
from main import load_and_save
load_and_save()
```

3. Run the main script:

```python
from main import main
main()
```

4. The application will prompt you to enter a question. Type your question and press Enter.
5. The system will retrieve relevant context, generate an answer using the ChatOllama model, and display the answer.
6. To exit, type exit and press Enter.



## Configuration
You can modify the following constants in the main.py file to adjust the behavior of the application:

* **CHAT_MODEL**: The name of the ChatOllama model to use (default: "wizardlm2").
* **EMBED_MODEL**: The name of the OllamaEmbeddings model to use (default: 'nomic-embed-text').
* **VECTOR_STORE_PATH**: The path to the vector store file (default: "vectorstore_pdf").
* **CHUNK_SIZE**: The maximum size of the text chunks (default: 100).
* **CHUNK_OVERLAP**: The overlap between adjacent text chunks (default: 50).

### Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)  for providing a powerful and flexible framework for building applications with large language models.
- [ChatOllama](https://github.com/ollamapub/chat-ollama)  for the open-source ChatOllama model used in this application.
- [Streamlit](https://streamlit.io/)  for the easy-to-use web app framework.