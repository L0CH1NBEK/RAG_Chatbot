from typing import Tuple

from fastapi.responses import JSONResponse
from typing import List
from config import EMBED_MODEL, TOP_K, CHROMA_DB_PATH, CHAT_MODEL, UPLOAD_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from core.loader import load_document
from core.chunker import chunk_documents
from core.utils import save_upload_to_disk
from core.vector_store import ChatRequest, build_or_replace_chroma_index

from fastapi import UploadFile, HTTPException, FastAPI, File


from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama


# def setup_knowledge(file_path):
#     docs = load_document(file_path)
#     chunks = chunk_documents(docs)
#     embeddings = get_embeddings()
#     vectorstore = create_vector_store(chunks, embeddings)
#     return vectorstore
#
# def chat_with_bot(vectorstore):
#     qa = build_rag_chain(vectorstore)
#     print("🤖 Ask me anything (type 'exit' to quit)")
#     while True:
#         query = input("You: ")
#         if query.lower() == "exit":
#             break
#         answer = qa.run(query)
#         print(f"Bot: {answer}")
#
app = FastAPI(title="RAG Chatbot (nomic-embed-text + Gemma3)")


@app.get("/")
def root():
    return {"message": "RAG Chatbot API (nomic-embed-text embeddings + Gemma3 chat). Endpoints: /upload [POST], /chat [POST]"}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        path = save_upload_to_disk(file, UPLOAD_DIR)
        docs = load_document(path)
        chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)
        if len(chunks) == 0:
            raise HTTPException(status_code=400, detail="No content extracted from file.")

        print(f"Loaded {len(docs)} docs → {len(chunks)} chunks. Building FAISS index…")
        _ = build_or_replace_chroma_index(chunks)
        return {"status": "ok", "message": f"Indexed {len(chunks)} chunks", "index_path": CHROMA_DB_PATH}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload/indexing failed: {e}")


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        top_k = req.top_k if req.top_k is not None else TOP_K
        answer = chat_rag(req.question, top_k=top_k)
        return JSONResponse({"answer": answer})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)



def build_prompt(context_blocks: List[str], question: str) -> List[dict]:
    context_text = "\n\n".join(
        [f"[CTX{i+1}]\n{blk}" for i, blk in enumerate(context_blocks)]
    )
    system = (
        "You are a multilingual assistant and ONLY responce from given context"
        "Detect the user’s input language (Uzbek or Russian). Always answer in the same language as the user’s question, even if the context is in another language"
        "If the user writes in Russian, answer in Russian."
        "If the user writes in Uzbek, answer in Uzbek."
        "Answer should be explicit informative. If question is not about context, say I apologize, but I'm trained to answer questions about given context."
    )

    user = (
        f"<CONTEXT>\n{context_text}\n</CONTEXT>\n\n"
        f"Question: {question}\n\n"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

def chat_rag(question: str, top_k: int = TOP_K) -> Tuple[str, List[str]]:
    # Load index and retrieve
    # query_lang = detect(question)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    store = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    retriever = store.as_retriever(
        search_kwargs={"k": top_k},
    )
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return (
            "Kontekst topilmadi. Avval hujjat yuklang. / Контекст не найден. Сначала загрузите документ.",
            [],
        )
    # Compose prompt
    context_blocks = [d.page_content for d in docs]
    messages = build_prompt(context_blocks, question)

    # Call Gemma3 via Ollama
    llm = ChatOllama(model=CHAT_MODEL, temperature=0.2)
    resp = llm.invoke(messages)
    answer = resp.content if hasattr(resp, "content") else str(resp)

    return answer
