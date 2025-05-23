# Conversational-PDF-RAG

Conversational-PDF-RAG is a Streamlit application that lets users chat with the contents of uploaded PDF documents using Retrieval-Augmented Generation (RAG). The app leverages LangChain, Ollama's LLaMA3.1 model, and Chroma for vector storage to provide concise, context-aware responses based on document content and chat history.

## 🚀 Features

- Upload one or multiple PDF files.
- Ask questions about the content of the PDFs.
- Maintains context with memory-enabled conversations.
- Reformulates follow-up questions for better retrieval.
- Uses LLaMA3.1 for high-quality natural language understanding.

## 🧠 Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **LLM**: [Ollama LLaMA3.1](https://ollama.com/)
- **Document Parsing**: LangChain's `PyPDFLoader`
- **Vector Store**: [Chroma](https://www.trychroma.com/)
- **Memory & Chains**: [LangChain](https://www.langchain.com/)

## 🚀 Getting Started

1. Clone the repository:
 ```
https://github.com/Saza-dev/LangChain-Ollama-Streamlit-LLaMA-3.1-Chatbot-Demo.git
 ```
2. Install Dependencies:
   Make sure you’re using Python 3.9 or later.
  ```
pip install -r requirements.txt
  ```
3. Install and Run Ollama with llama 3.1
```
ollama run llama3.1
```
4. Run the App
```
streamlit run app.py
```
