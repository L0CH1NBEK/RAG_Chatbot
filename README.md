# Chatbot with RAG and Gemma3

This project implements a Retrieval-Augmented Generation (RAG) chatbot using:
- **Chroma** as the vector database
- **nomic-embed-text** from Ollama for embeddings
- **Gemma3 LLM** for answering questions

The chatbot has two main APIs:
1. **Upload API** – Upload documents, chunk them, embed with nomic-embed-text, and store in Chroma DB.
2. **Chat API** – Query the chatbot; relevant chunks are retrieved from Chroma and passed to Gemma3 for generating answers.

---

## 📂 Project Structure

```
.
├── app.py                # FastAPI entry point (defines APIs)
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── core/
│   ├── chunker.py        # Splits documents into chunks
│   ├── loader.py         # Loads documents from /data
│   ├── utils.py          # Helper utilities
│   └── vector_store.py   # Handles Chroma DB operations
├── data/                 # Uploaded documents storage
└── chroma_db/            # Persistent Chroma vector database
```

---

## ⚙️ Setup

### 1. Clone repository & create venv
```bash
git clone https://github.com/your-repo/chatbot-rag.git
cd chatbot-rag
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate    # Windows
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Ollama (for embeddings & LLM)
Ensure you have [Ollama](https://ollama.ai) installed and running.

Pull required models:
```bash
ollama pull nomic-embed-text
ollama pull gemma3
```

### 4. Run the FastAPI app
```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

---

## 🚀 API Endpoints

### 1. Upload Document
**POST** `/upload`  
Uploads a document, chunks it, embeds with `nomic-embed-text`, and stores in Chroma DB.

Request (multipart/form-data):
```
file=@example.pdf
```

Response:
```json
{"message": "File uploaded and indexed successfully."}
```

---

### 2. Chat with Bot
**POST** `/chat`  
Ask a question, retrieve knowledge from Chroma, and generate an answer with `gemma3`.

Request:
```json
{
  "query": "How I can use this website?"
}
```

Response:
```json
{
  "answer": "This website is for..."
}
```

---

## 📌 Notes
- Embeddings are stored persistently inside `chroma_db/`.
- Uploaded documents remain in `data/`.
- Multi-language support depends on LLM capability (`gemma3` supports multiple languages).

---

## 🛠 Future Improvements
- Add authentication for APIs  
- Improve chunking strategy  
- Support streaming responses  
- Add language detection + translation in RAG pipeline  

---

## 🧑‍💻 Author
Developed by L0CH1NBEK
