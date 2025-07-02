from typing import List
import os

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from chunker import chunk_documents
from cleaner import clean_chunks
from chroma_store import store_chunks

from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

# TODO: Error handling for non-parseable text files.


class PDFParsingError(Exception):
    """Raised when PDF parsing fails."""

def parse_pdf(file_path: str) -> List[Document]:
    """
    Load and parse a PDF into LangChain Document objects using PyMuPDFLoader.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of Document objects, each containing page_content and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        PDFParsingError: If parsing fails for any other reason.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        raise PDFParsingError(f"Failed to parse PDF '{file_path}': {e}") from e
    

"""Returns Document file with metadata linked into itself
result[0].metadata output
{'producer': 'pdfTeX-1.40.25', 'creator': 'LaTeX with hyperref', 'creationdate': '2024-04-10T21:11:43+00:00',
  'source': '../database/attention-paper.pdf', 'file_path': '../database/attention-paper.pdf', 'total_pages': 15, 
  'format': 'PDF 1.5', 'title': '', 'author': '', 'subject': '', 'keywords': '', 
  'moddate': '2024-04-10T21:11:43+00:00',
  'trapped': '', 'modDate': 'D:20240410211143Z', 'creationDate': 'D:20240410211143Z', 'page': 13}

  ===> Each page has it's own metadata along with page number and file name linked with it.
  ===> Now, it's time to find out whether these metadata are embedded with each chunk data.
"""


result = parse_pdf("../database/illusion.pdf")
result_chunk = chunk_documents(result)
clean_chunk = clean_chunks(result_chunk)
vector_store = store_chunks(clean_chunk)

model = ChatOllama(
    model="gemma3:4b"
)

retriever = vector_store.as_retriever()

SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_TEMPLATE),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

document_chain = create_stuff_documents_chain(model, question_answering_prompt)

def parse_retriever_input(params):
    # Extract the latest user message content
    return params["messages"][-1].content

retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever,
).assign(
    answer=document_chain,
)

chat_history = []

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    chat_history.append(HumanMessage(content=user_input))
    response = retrieval_chain.invoke({"messages": chat_history})
    answer = response["answer"]

    print("AI:", answer)
    chat_history.append(AIMessage(content=answer))
