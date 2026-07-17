from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic_models import (
    QueryInput,
    QueryResponse,
    DocumentInfo,
    DeleteFileRequest,
)

from langchain_utils import get_rag_chain

from db_utils import (
    insert_application_logs,
    get_chat_history,
    get_all_documents,
    insert_document_record,
    delete_document_record,
)

from chroma_utils import (
    index_document_to_chroma,
    delete_doc_from_chroma,
)

import uuid
import shutil
import os
import logging


logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
)


app = FastAPI(
    title="Conversational RAG Chatbot"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def home():
    return {
        "message": "Conversational RAG Chatbot API is Running..."
    }



# ---------------- CHAT ----------------

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):

    try:

        session_id = query_input.session_id or str(uuid.uuid4())


        logging.info(
            f"Session: {session_id} | Question: {query_input.question}"
        )


        chat_history = get_chat_history(session_id)


        rag_chain = get_rag_chain(
            query_input.model.value
        )


        result = rag_chain.invoke(
            {
                "input": query_input.question,
                "chat_history": chat_history,
            }
        )


        answer = result["answer"]


        # FIXED HERE
        insert_application_logs(
            session_id=session_id,
            user_query=query_input.question,
            assistant_response=answer,
            model=query_input.model.value,
        )


        return QueryResponse(
            answer=answer,
            session_id=session_id,
            model=query_input.model,
        )


    except Exception as e:

        logging.exception(e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )



# ---------------- UPLOAD DOCUMENT ----------------

@app.post("/upload-doc")
def upload_document(file: UploadFile = File(...)):

    allowed_extensions = [
        ".pdf",
        ".docx",
        ".html"
    ]


    extension = os.path.splitext(
        file.filename
    )[1].lower()


    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
        )


    temp_file = f"temp_{file.filename}"


    try:

        with open(temp_file, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )


        file_id = insert_document_record(
            file.filename
        )


        success = index_document_to_chroma(
            temp_file,
            file_id
        )


        if success:

            return {
                "message": "Document Indexed Successfully",
                "file_id": file_id
            }


        delete_document_record(file_id)


        raise HTTPException(
            status_code=500,
            detail="Failed to index document"
        )


    finally:

        if os.path.exists(temp_file):

            os.remove(temp_file)



# ---------------- LIST DOCUMENTS ----------------

@app.get(
    "/list-docs",
    response_model=list[DocumentInfo]
)
def list_documents():

    return get_all_documents()



# ---------------- DELETE DOCUMENT ----------------

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):


    chroma_success = delete_doc_from_chroma(
        request.file_id
    )


    if not chroma_success:

        raise HTTPException(
            status_code=500,
            detail="Unable to delete from ChromaDB"
        )


    db_success = delete_document_record(
        request.file_id
    )


    if not db_success:

        raise HTTPException(
            status_code=500,
            detail="Deleted from Chroma but database delete failed"
        )


    return {
        "message": "Document Deleted Successfully"
    }