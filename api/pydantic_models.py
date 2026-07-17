from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    GROQ_LLAMA_3_1 = "llama-3.1-8b-instant"
    GROQ_LLAMA_3_3 = "llama-3.3-70b-versatile"


class QueryInput(BaseModel):
    question: str
    session_id: str | None = None
    model: ModelName = Field(default=ModelName.GROQ_LLAMA_3_1)


class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName


class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime


class DeleteFileRequest(BaseModel):
    file_id: int