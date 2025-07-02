from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# Creating base & embedding model object
model=ChatOllama(
    model="gemma3:4b"
)

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest"
)


# Parse the pdf using PyMuPDF loader
file_path="./database/Shiven_Resume.pdf"
loader = PyMuPDFLoader(file_path)
pdf_pages = loader.load()

# Convert the Parsed pdf pages data to small chunks
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
    )

chunks_collection = text_splitter.split_documents(pdf_pages)

# Clean the chunks to remove redundant newline characters from page_content attribute
chunks_refined = []
for chunk in chunks_collection:
    cleaned_content = chunk.page_content.replace("\n", " ")
    chunks_refined.append(Document(page_content=cleaned_content, metadata=chunk.metadata))

# Store the refined chunks inside a vector store chromadb
store = Chroma.from_documents(
    chunks_refined,
    embedding_model,
    collection_name="paper_collection"
)

# Creating a retriever to invoke relevant docs
retriever = store.as_retriever(
    search_kwargs={
        "k" : 4
        }
    )

# Create a complete RAG chain
template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert assistant. Answer the question using only the provided context."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Context: {context}\n\nQuestion: {question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
        "chat_history": lambda x: []  # You can expand this for multi-turn conversations
    }
    | template
    | model
)

while True:
    user_input = input("User => ")
    if user_input == "exit":
        break
    response = rag_chain.invoke(user_input)
    print(response.content)

