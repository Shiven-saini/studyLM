from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def build_vector_store(chunked_docs_list, persist_dir="./chroma_store", collection_name="research_paper_collection"):
    """
        Take chunked docs list as an input and perform vectorization.
        Returns the vector_store object for furthe processing and retrieval strategy.the 
        TODO: pull out the model name and embeddings engine from the function for better modularity.
    """
    
    
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vector_store = Chroma.from_documents(
        chunked_docs_list,
        embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name
    )

    return vector_store
