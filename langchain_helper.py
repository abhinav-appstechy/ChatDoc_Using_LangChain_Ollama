# Data Ingestion
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


def data_ingestion_for_txt_file(data_file):
    loader = TextLoader(data_file)
    txt_document = loader.load()
    return txt_document
    

def text_splitter(doc):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(doc)
    return documents

def vector_embedding_using_chromadb(doc):
    # Vector Embeddings And Vector Store
    db = Chroma.from_documents(doc, OllamaEmbeddings(model="gemma:2b"))
    return db


def llm_model():
    # Load Ollama Gemma model
    llm = Ollama(model="gemma:2b")
    return llm


def chat_prompt_template(input):
    prompt = ChatPromptTemplate.from_template("""
    Answer the following questions based only on the provided context.
    Think step by step before providing a detailed answer.
    I will tip you $1000 if the user finds the answer helpful.
    <context>
    {context}
    </context>
    Question: {input}
    """)
    return prompt


def document_chaining(llm, prompt):
    document_chain = create_stuff_documents_chain(llm, prompt)
    return document_chain


def retriever_chain(db, document_chain):
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def result(retrieval_chain, input):
    response = retrieval_chain.invoke({"input":input})
    return response["answer"]


def main(text_document_txt, prompt ):
    document = text_splitter(text_document_txt)
    db = vector_embedding_using_chromadb(document)
    llm = llm_model()
    chat_prompt = chat_prompt_template(prompt)
    document_chain = document_chaining(llm, chat_prompt)
    retriever_chaining = retriever_chain(db, document_chain)
    final_result = result(retriever_chaining, prompt)
    return final_result