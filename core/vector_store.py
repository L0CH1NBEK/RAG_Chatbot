from typing import Optional

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel
from typing import List
from config import EMBED_MODEL, CHROMA_DB_PATH, BATCH_SIZE


class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = None



def batch_embed_texts(emb: OllamaEmbeddings, texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[float]]:
    """Embed texts in smaller batches to avoid long stalls."""
    vectors: List[List[float]] = []
    total = len(texts)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]
        # print progress to server logs
        print(f"Embedding batch {start//batch_size + 1}/{(total + batch_size - 1)//batch_size} ({len(batch)} items)")
        vecs = emb.embed_documents(batch)
        vectors.extend(vecs)
    return vectors


def build_or_replace_chroma_index(chunks: List[Document]):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    print('Embeddings loaded')
    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    print('Vectors created')
    vectordb.persist()
    return {"status": "ok", "message": "Document stored successfully."}
