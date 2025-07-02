from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text:latest"
    )

response = embeddings.embed_query("Hello world")
print(len(response))

response = embeddings.embed_documents(["Hello", " world"])
print(len(response[0]) + len(response[1]))