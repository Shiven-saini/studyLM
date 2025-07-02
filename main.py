from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from PyPDF2 import PdfReader

# # Load the test content
# with open("test.txt") as f:
#     test_content = f.read()

pdf_path = "Shiven-Resume.pdf"
reader = PdfReader(pdf_path)

test_content = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        test_content += text + "\n"  # Add newline between pages

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=100,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False
# )

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents([test_content])

file = open("test-split.txt", "w")

for i in texts:
    file.write(i.page_content + '\n')

file.close()


embedding_model=OllamaEmbeddings(model="nomic-embed-text:latest")

vector_store = Chroma.from_documents(texts, embedding_model)
chat_model = ChatOllama(model="gemma3:4b")

while True:
    query = input("Enter your question (or 'quit' to exit): ")
    if query.lower() == "quit":
        print("Exiting.")
        break

    docs = vector_store.similarity_search(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    messages = [
        ("system", "You are a helpful assistant."),
        ("user", f"Context:\n{context}\n\nQuestion: {query}")
    ]

    response = chat_model.invoke(messages)
    print("Answer:", response.content)
    print()


