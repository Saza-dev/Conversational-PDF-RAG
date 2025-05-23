# libraries
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# embeddings
embeddings = OllamaEmbeddings(model="llama3.1")

# setup streamlit app
st.title("Conversational RAG with PDF upload and chat history")
st.write("Upload Pdf's and chat with their content")

# model 
llm = ChatOllama(model="llama3.1")

# chat interface
session_id = st.text_input("Session ID",value="default_session")

# manage the chat history
if 'store' not in st.session_state:
    st.session_state.store={}

uploaded_files = st.file_uploader("Choose a PDF file",type="pdf",accept_multiple_files=True)

# Process uploaded files
if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        tempPdf = f"./temp.pdf"
        with open(tempPdf,'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name
        
        loader=PyPDFLoader(tempPdf)
        docs=loader.load()
        documents.extend(docs)

    # split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
    splits = text_splitter.split_documents(documents) 
    vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
    retriver = vectorstore.as_retriever()

    # system prompt
    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "Which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    # chat prompt template
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )

    # history_aware_retriver, Reformulate context-dependent user questions into standalone questions
    history_aware_retriver = create_history_aware_retriever(llm,retriver,contextualize_q_prompt)

    # Answer questions
    # System prompt for answering questions
    system_prompt = (
        "You are an assistant for question answering tasks. "
        "Use the following pieces of retrived context to answer "
        "the question. If you dont know the answer say that you "
        "dont't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    # ChatPromptTemplate for the QA task
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )

    # Create a QA chain using the above prompt
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)

    # Create the full Retrieval-Augmented Generation (RAG) chain
    rag_chain=create_retrieval_chain(history_aware_retriver,question_answer_chain)


    # Function to get or create a session-specific chat history
    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]

    # Create a Runnable chain that automatically manages message history
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Your question:")
    if user_input:
        session_history=get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input":user_input},
            config={
                "configurable":{"session_id":session_id}
            }
        )
        st.write(st.session_state.store)
        st.write("Assistant",response['answer'])
        st.write("Chat History:",session_history.messages)


