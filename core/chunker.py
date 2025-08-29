from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from config import CHUNK_OVERLAP, CHUNK_SIZE


def chunk_documents(docs: List[Document], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

