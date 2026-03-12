# AI Teacher Malaysia — RAG System
### AI Guru Matematik KSSM Tingkatan 1–5

A Retrieval-Augmented Generation (RAG) AI tutor trained on official Malaysian KSSM Math textbooks. Students can ask questions in **Bahasa Malaysia or English** and receive step-by-step answers grounded in their actual syllabus.

---

## Architecture

```
KSSM PDFs (T1-T5)
    ↓ EasyOCR + PyMuPDF
Text Chunks + Embeddings
    ↓ sentence-transformers
Zilliz Cloud (Vector DB)
    ↓ Similarity Search
FastAPI Backend (RAG Chain)
    ↓ Grok API (grok-2-1212)
Bilingual Answer (BM / EN)
    ↓
Web Chat UI (HTML/CSS/JS)
```

---

## Quick Setup

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
# Copy the template
copy .env.example .env
```
Open `.env` and fill in:
- `ZILLIZ_URI` — from [cloud.zilliz.com](https://cloud.zilliz.com) → Create Cluster → Copy endpoint
- `ZILLIZ_TOKEN` — from Zilliz Cloud → API Keys
- `GROK_API_KEY` — from [console.x.ai](https://console.x.ai)

### 3. Run PDF Ingestion (One Time)
```bash
# Dry run first to test PDF reading
python ingest_run.py --dry-run

# Full ingestion → uploads to Zilliz Cloud
python ingest_run.py
```
⏱️ *Ingestion takes ~30-60 min for all 5 textbooks depending on CPU/GPU.*

### 4. Start the Server
```bash
python start.py
```
Or use the batch file:
```
START_SERVER.bat
```

### 5. Open the App
Visit: **http://localhost:8000**

---

## File Structure

```
Ai teacher Malaysia/
├── backend/
│   ├── config.py         # Configuration (env vars)
│   ├── ingest.py         # PDF parsing + EasyOCR + embeddings
│   ├── vector_store.py   # Zilliz Cloud operations
│   ├── llm.py            # Grok API wrapper (streaming)
│   ├── rag_chain.py      # RAG orchestration (bilingual)
│   └── api.py            # FastAPI endpoints
├── frontend/
│   ├── index.html        # Chat UI
│   ├── styles.css        # Premium dark design
│   └── app.js            # SSE streaming + MathJax
├── T1_Math_BukuTeks_KSSM.pdf
├── T2_Math_BukuTeks_KSSM.pdf
├── T3_Math_BukuTeks_KSSM.pdf
├── T4_Math_BukuTeks_KSSM.pdf
├── T5_Math_BukuTeks_KSSM.pdf
├── ingest_run.py         # CLI ingestion script
├── start.py              # Server launcher
├── requirements.txt      # Python dependencies
├── .env.example          # Config template
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | System health + chunk count |
| `POST` | `/api/chat` | Student chat (SSE streaming) |
| `POST` | `/api/ingest` | Admin: trigger re-ingestion |
| `GET` | `/` | Web chat interface |

### Chat Request Format
```json
POST /api/chat
{
  "question": "Selesaikan: 2x + 5 = 13",
  "language": "bm",
  "form_filter": "T2"
}
```
- `language`: `"bm"` | `"en"` | `"auto"` (default: auto-detect)
- `form_filter`: `"T1"` to `"T5"` or omit for all forms

---

## Features

- 📚 **KSSM-grounded** — answers from official Tingkatan 1–5 Math textbooks
- 🌏 **Bilingual** — Bahasa Malaysia & English
- 🔢 **Step-by-step math** — numbered solution steps
- 🖼️ **EasyOCR** — handles image-based/scanned PDF pages
- ⚡ **Streaming** — real-time token-by-token response
- 🧮 **MathJax** — renders equations in the browser
- 🎓 **Form filter** — restrict answers to specific form level

---

## Zilliz Cloud Setup

1. Go to [cloud.zilliz.com](https://cloud.zilliz.com) → Sign up (free)
2. Create Cluster → Select **Free Tier** → Region: any
3. Copy **Public Endpoint URI** → paste as `ZILLIZ_URI` in `.env`
4. Go to **API Keys** → Create key → paste as `ZILLIZ_TOKEN`

---

## Grok API Setup

1. Go to [console.x.ai](https://console.x.ai) → Sign up
2. Create API key → paste as `GROK_API_KEY` in `.env`
3. Model `grok-2-1212` is used by default (configurable via `GROK_MODEL`)
