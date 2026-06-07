import os
import glob
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings  #The entire section is used to import necessary libraries and set up the environment for the script. It includes loading environment variables, defining paths for the vector database and knowledge base, and initializing the embeddings model. from number 1 its the os and the glob library to handle file paths and searching for files, from pathlib to manage file paths in a more convenient way, and from dotenv to load environment variables. The langchain libraries are used for document loading, text splitting, vector store management, and embeddings generation.

load_dotenv(override=True)  # Load environment variables from a .env file, overriding existing ones if necessary.

DB_NAME = str(Path(__file__).parent.parent / "vector_db") # Define the path for the vector database, which will be used to store the embeddings generated from the documents in the knowledge base. The path is set to a folder named "vector_db" located in the parent directory of the current file's directory.
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base") # Define the path for the knowledge base, which is where the markdown files containing the information to be ingested are stored. The path is set to a folder named "knowledge-base" located in the parent directory of the current file's directory.

embeddings = OpenAIEmbeddings(model="text-embedding-3-large") # Initialize the OpenAIEmbeddings model with the specified model name "text-embedding-3-large". This model will be used to generate embeddings for the documents in the knowledge base, which will then be stored in the vector database for later retrieval and use in a RAG (Retrieval-Augmented Generation) chatbot.


def fetch_documents():
    """Load markdown files from knowledge base folders."""
    documents = []
    for folder in glob.glob(str(Path(KNOWLEDGE_BASE) / "*")):
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        for doc in loader.load():
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents #this entire function is responsible for loading markdown files from the specified knowledge base folders. It uses the glob library to search for all folders within the knowledge base directory, and for each folder, it initializes a DirectoryLoader to load all markdown files (with a .md extension) using the TextLoader class. The loaded documents are stored in a list, and each document's metadata is updated to include the type of document based on the folder it was loaded from. Finally, the function returns the list of loaded documents. 1st the function initializes an empty list called documents to store the loaded documents. It then uses glob to search for all folders within the knowledge base directory, and for each folder found, it extracts the folder name to determine the document type. A DirectoryLoader is created for each folder, specifying that it should load all markdown files using the TextLoader class with UTF-8 encoding. The loaded documents are appended to the documents list, and their metadata is updated to include the document type. Finally, the function returns the complete list of loaded documents.


def create_chunks(documents):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    return splitter.split_documents(documents) # This function takes a list of documents as input and splits them into smaller, overlapping chunks using the RecursiveCharacterTextSplitter. The chunk_size parameter specifies the maximum size of each chunk (in characters), while the chunk_overlap parameter specifies how much overlap there should be between consecutive chunks. The function returns a list of the resulting chunks, which can then be used for further processing, such as generating embeddings and storing them in a vector database. It first splits the input documents into smaller chunks based on the specified chunk size and overlap. The RecursiveCharacterTextSplitter is used to ensure that the chunks are created in a way that maintains the context of the original documents, allowing for better retrieval and understanding when used in a RAG chatbot. The resulting list of chunks is returned for further processing.


def build_vectorstore(chunks):
    """Create or overwrite Chroma vectorstore."""
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )

    count = vectorstore._collection.count()
    dims = len(vectorstore._collection.get(limit=1, include=["embeddings"])["embeddings"][0])
    print(f"✅ Vectorstore ready: {count:,} vectors, {dims:,} dimensions")
    return vectorstore #This function is responsible for creating or overwriting a Chroma vectorstore using the provided chunks of text. It first checks if a vectorstore already exists at the specified DB_NAME path, and if it does, it deletes the existing collection to ensure that the new data will be stored without conflicts. Then, it creates a new vectorstore from the provided chunks using the Chroma.from_documents method, which takes the chunks, the embeddings model, and the persist directory as arguments. After creating the vectorstore, it retrieves and prints the count of vectors and their dimensions to confirm that the vectorstore is ready for use. Finally, it returns the created vectorstore for further use in the application. It first checks if a vectorstore already exists at the specified path, and if it does, it deletes the existing collection to ensure that the new data will be stored without conflicts. Then, it creates a new vectorstore from the provided chunks using the Chroma.from_documents method, which takes the chunks, the embeddings model, and the persist directory as arguments. After creating the vectorstore, it retrieves and prints the count of vectors and their dimensions to confirm that the vectorstore is ready for use. Finally, it returns the created vectorstore for further use in the application. Then, it creates a new vectorstore from the provided chunks using the Chroma.from_documents method, which takes the chunks, the embeddings model, and the persist directory as arguments. After creating the vectorstore, it retrieves and prints the count of vectors and their dimensions to confirm that the vectorstore is ready for use. Finally, it returns the created vectorstore for further use in the application.


if __name__ == "__main__":
    # This block runs only when the script is executed directly (python ingest2.py).
    # It will not run when the file is imported as a module from another script.
    # Steps performed:
    # 1. fetch_documents(): load markdown files from the knowledge-base folders.
    # 2. create_chunks(docs): split the loaded documents into overlapping text chunks.
    # 3. build_vectorstore(chunks): create/overwrite the Chroma vector store with embeddings.
    # Finally, print a completion message.
    docs = fetch_documents()
    chunks = create_chunks(docs)
    build_vectorstore(chunks)
    print("🚀 Ingestion complete")
