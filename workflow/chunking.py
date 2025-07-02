# Module responsible for loading and chunking of data.
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    chunked_docs_list = splitter.split_documents(docs)

    return chunked_docs_list


docs = load_and_chunk_pdf("../attention-paper.pdf")
print(docs)