import os
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "data")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")  # from Ollama
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemma3")     # from Ollama
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "100"))
TOP_K = int(os.environ.get("TOP_K", "4"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
CHROMA_DB_PATH = "./chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)