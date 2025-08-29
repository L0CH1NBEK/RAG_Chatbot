from http.client import HTTPException

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from typing import List

def load_document(path: str) -> List[Document]:
    if path.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif path.lower().endswith(".docx"):
        loader = Docx2txtLoader(path)
    elif path.lower().endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT.")
    return loader.load()
