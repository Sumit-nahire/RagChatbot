from dotenv import load_dotenv
load_dotenv()

from typing import List

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_chroma import Chroma



# -----------------------------
# Text Splitter
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)



# -----------------------------
# HuggingFace Embeddings
# -----------------------------
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



# -----------------------------
# Chroma Vector Store
# -----------------------------
vectorstore = Chroma(
    collection_name="documents",
    persist_directory="./chroma_db",
    embedding_function=embedding_function,
)



# -----------------------------
# Load & Split Document
# -----------------------------
def load_and_split_document(file_path: str) -> List[Document]:

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)

    elif file_path.endswith(".html"):
        loader = UnstructuredHTMLLoader(file_path)

    else:
        raise ValueError(f"Unsupported file type: {file_path}")


    documents = loader.load()

    splits = text_splitter.split_documents(documents)

    return splits



# -----------------------------
# Add Document To Chroma
# -----------------------------
def index_document_to_chroma(file_path: str, file_id: int):

    try:

        splits = load_and_split_document(file_path)

        for split in splits:
            split.metadata["file_id"] = file_id


        vectorstore.add_documents(splits)

        return True


    except Exception as e:

        print("Index Error :", e)

        return False



# -----------------------------
# Delete Document From Chroma
# -----------------------------
def delete_doc_from_chroma(file_id: int):

    try:

        vectorstore._collection.delete(
            where={
                "file_id": file_id
            }
        )

        return True


    except Exception as e:

        print("Delete Error :", e)

        return False