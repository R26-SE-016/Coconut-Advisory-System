# Coconut Advisory System - FastAPI Backend

A high-performance FastAPI backend service powering **SaruPol (CocoCastAI)**, a Retrieval-Augmented Generation (RAG) advisory system designed for Sri Lankan coconut farmers.

It provides RAG knowledge retrieval from Coconut Research Institute (CRI) guidelines, Multi-LLM consensus validation, high-accuracy agricultural Sinhala translation, and Neural Text-to-Speech streaming.

---

## 🌟 Key Features

- **Knowledge Base RAG Engine**: Uses FAISS vector search with `sentence-transformers/all-MiniLM-L6-v2` embeddings over CRI coconut cultivation documents.
- **Multi-LLM Validator & Jury Judge (`/ask-multi`)**:
  - Queries candidate LLMs in parallel (`LLaMA 3.3 70B`, `LLaMA 3.1 8B`, `Qwen`).
  - Employs an AI Judge model (`openai/gpt-oss-120b`) to evaluate responses, output a consensus score, and select the best advisory answer.
  - Automatically translates all 5 response fields (`best_answer`, `reason`, `llama_answer`, `llama8b_answer`, `qwen_answer`) into natural Sinhala when `language == "si"`.
- **Farmer-Friendly Sinhala Translation Engine**:
  - High-accuracy model cascade (`openai/gpt-oss-120b` $\rightarrow$ `llama-3.1-8b-instant`).
  - Specialized Sri Lankan coconut extension terminology enforcement (e.g. `Wet Zone` $\rightarrow$ `තෙත් කලාපය`, `young coconut palms` $\rightarrow$ `තරුණ පොල් ගස්`).
  - Automatic regex sanitizers to strip reasoning `<think>` tags and fix hallucinated terms.
- **Server-Side Context Resolution**: Automatically determines Sri Lankan agro-climatic zones (Wet, Intermediate, Dry) from GPS coordinates and system agricultural seasons (Yala / Maha).
- **Neural Text-To-Speech (`/tts`)**:
  - Streams audio in real-time using `si-LK-SameeraNeural` for Sinhala and `en-US-AriaNeural` for English.
  - Includes text normalization for Sinhala conjunct consonants and letter-by-letter pronunciation hints for fertilizer codes (e.g., `YPM-W`).

---

## 🛠️ Tech Stack

- **Framework**: FastAPI / Uvicorn / Gunicorn
- **Embeddings & Vector Store**: SentenceTransformers & FAISS
- **LLM Provider**: Groq API (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`)
- **TTS Engine**: `edge-tts` (Microsoft Edge Neural Speech)
- **Language**: Python 3.9+

---

## 🚀 Quick Setup & Installation

### Prerequisites

- Python 3.9 or higher
- Groq API Key ([Get one here](https://console.groq.com/))
- FAISS vector index generated via `step1_build_index.py`

### 1. Environment Setup

```bash
cd backend
python -m venv venv

# Activate Virtual Environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create `.env` inside `backend/` directory:

```env
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Required Groq API Key
GROQ_API_KEY=your_groq_api_key_here

# Directory Paths
KNOWLEDGE_BASE_DIR=../knowledge_base
FAISS_INDEX_DIR=../faiss_index

# CORS Origins
ALLOWED_ORIGINS=*
```

### 3. Build Vector Index (First Time Only)

If `faiss_index/` is not yet generated, build it from your knowledge base PDFs:

```bash
cd ..
python step1_build_index.py
```

### 4. Start Server

```bash
cd backend
python -m app.main
```

The backend server will run at: `http://localhost:8000`  
Swagger API Documentation: `http://localhost:8000/docs`

---

## 📖 API Endpoints Reference

### 1. Standard RAG Advisory Query (`POST /ask`)

Queries the RAG knowledge base for coconut farming advice.

- **URL**: `/ask`
- **Method**: `POST`
- **Request Body**:
```json
{
  "question": "How should I fertilize young coconut palms?",
  "context": "Wet Zone | Yala Season (August)",
  "language": "si"
}
```
- **Response**:
```json
{
  "success": true,
  "question": "තරුණ පොල් ගස් වලට පොහොර යෙදිය යුත්තේ කෙසේද?",
  "answer": "තරුණ පොල් ගස් සඳහා තෙත් කලාපයේ YPM-W පොහොර මිශ්‍රණය භාවිතා කිරීමට නිර්දේශ කරමි...",
  "sources": [
    {
      "title": "English.pdf",
      "content": "• Wet Zone: Rainfall in the wet zone...",
      "metadata": { "source": "English.pdf" }
    }
  ],
  "confidence": 0.85,
  "context_used": "Wet Zone | Yala Season (August)"
}
```

---

### 2. Multi-LLM Consensus Validator (`POST /ask-multi`)

Queries 3 candidate models, uses an AI Judge to select the best response, and translates all candidate answers into Sinhala if requested.

- **URL**: `/ask-multi`
- **Method**: `POST`
- **Request Body**:
```json
{
  "question": "How do I control termites in coconut nursery?",
  "latitude": 6.9271,
  "longitude": 79.8612,
  "language": "si"
}
```
- **Response**:
```json
{
  "success": true,
  "best_answer": "පොල් තවානේ වේයන් පාලනය කිරීම සඳහා...",
  "best_model": "llama8b",
  "reason": "LLaMA 3.1 8B පිළිතුර CRI උපදෙස් වලට වඩාත් ගැළපෙන අතර සරල පියවර ලබා දෙයි.",
  "consensus_score": 85,
  "llama_answer": "පොල් තවාන් වල වේයන් පාලන පියවර...",
  "llama8b_answer": "පොල් තවානේ වේයන් පාලනය කිරීම සඳහා...",
  "qwen_answer": "තවානේ වේයන් හානිය පාලනය කිරීමට...",
  "sources": [...],
  "zone": "Wet Zone",
  "season": "Yala (August)"
}
```

---

### 3. Batch Translation (`POST /translate-batch`)

Translates a list of chat items into the target language (`si` or `en`).

- **URL**: `/translate-batch`
- **Method**: `POST`
- **Request Body**:
```json
{
  "messages": [
    { "id": "msg1", "text": "How do I select a good mother palm?" },
    { "id": "msg2", "text": "What is the recommended spacing?" }
  ],
  "target_lang": "si"
}
```
- **Response**:
```json
{
  "success": true,
  "translations": [
    { "id": "msg1", "translated_text": "හොඳ මව් පොල් ගසක් තෝරා ගන්නේ කෙසේද?" },
    { "id": "msg2", "translated_text": "පොල් ගස් සිටුවීමට නිර්දේශිත පරතරය කුමක්ද?" }
  ]
}
```

---

### 4. Text-To-Speech Stream (`GET /tts`)

Generates an audio stream of the given text in natural neural speech.

- **URL**: `/tts?text=තරුණ පොල් ගස් වලට පොහොර යෙදීම&lang=si`
- **Method**: `GET`
- **Response**: Audio stream (`audio/mpeg`)

---

### 5. Health & Info Endpoints

- **`GET /health`**: Returns system health and status of RAG retriever loading.
- **`GET /info`**: Returns API metadata and registered endpoints.

---

## 📂 Project Structure

```
coconut_advisory_system/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI entry point & API endpoints
│   ├── requirements.txt         # Dependencies
│   ├── .env.example             # Environment template
│   └── README.md                # Backend documentation
├── faiss_index/                 # FAISS vector store
├── knowledge_base/              # PDF documents (CRI guidelines)
├── step1_build_index.py         # Document embedding & vector indexing
└── step2_rag_engine.py          # Core RAG retrieval & Sinhala translator
```

---

## 🛡️ License

Developed for the Coconut Advisory System (SaruPol / CocoCastAI).
