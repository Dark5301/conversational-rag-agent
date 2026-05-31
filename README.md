# 🧠 Conversational RAG Agent

A production-grade, conversational AI assistant built with **PydanticAI**, **Qdrant Cloud**, and **LogFire**. This project implements a full Retrieval-Augmented Generation (RAG) pipeline with semantic caching, context window compression, and structured observability — all wired together into a single, clean orchestration layer.

> Built on *The Gift of the Magi* by O. Henry as the demo knowledge base, but easily adaptable to any document corpus.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **Semantic RAG** | Retrieves the most relevant document chunks from Qdrant Cloud using cosine similarity |
| ⚡ **Semantic Cache** | Skips the LLM entirely on similar repeated questions using vector-based cache lookup |
| 🗜️ **Context Compression** | A dedicated Summarizer Agent compresses old conversation history to stay within token limits |
| 📊 **Full Observability** | Every span, latency, cache hit, token cost, and error is tracked via LogFire |
| 🛡️ **Guardrailed Responses** | The main agent is strictly instructed to only answer from retrieved context — no hallucination |
| 🔄 **Idempotent Ingestion** | Document ingestion is safe to re-run; uses deterministic MD5-based UUIDs for deduplication |

---

## 🏗️ Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────┐
│         ConversationOrchestrator        │
│                                         │
│  1. Embed user input                    │
│  2. Check Semantic Cache (Qdrant)  ──── ── ─► Cache HIT → Return instantly
│  3. [Cache MISS] Retrieve RAG chunks    │
│  4. Build prompt (history + context)    │
│  5. Run Main Agent (gpt-4o-mini)        │
│  6. Save response to Cache              │
│  7. Update conversation history         │
│  8. [If history > limit] Compress via   │
│     Summarizer Agent                    │
└─────────────────────────────────────────┘
    │
    ▼
 Response
```

### Components

```
project2.py        ← Main orchestrator: ties everything together
RAG.py             ← Document ingestion, chunking, embedding, and retrieval
semantic_cache.py  ← Vector-based semantic cache backed by Qdrant Cloud
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A [Qdrant Cloud](https://cloud.qdrant.io/) account (free tier works)
- An OpenAI-compatible API key (this project uses [aicredits.in](https://aicredits.in), but any OpenAI-compatible endpoint works)
- [LogFire](https://logfire.pydantic.dev/) account for observability

### 1. Clone the Repository

```bash
git clone https://github.com/Dark5301/conversational-rag-agent.git
cd conversational-rag-agent
```

### 2. Install Dependencies

```bash
pip install pydantic-ai openai qdrant-client logfire python-dotenv
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```env
# Your OpenAI-compatible API key and base URL
AICREDITS_API_KEY=your_api_key_here

# Qdrant Cloud credentials
QDRANT_URL=https://your-cluster-url.qdrant.tech
QDRANT_API_KEY=your_qdrant_api_key_here
```

### 4. Prepare Your Document

Update the `filepath` in `RAG.py → load_documents()` to point to your own `.txt` file. The current default is:

```python
filepath = '/path/to/your/document.txt'
```

If you're using a Project Gutenberg text (like this demo does), the chunker will automatically strip the header/footer boilerplate using the standard `*** START OF ... ***` / `*** END OF ... ***` markers.

### 5. Run Document Ingestion

> ⚠️ **Run this exactly once.** It parses your document, generates embeddings, and uploads vectors to Qdrant Cloud. Re-running is safe due to deterministic UUIDs, but unnecessary.

```bash
python3 RAG.py
```

You'll see LogFire spans confirming the ingestion pipeline: file load → chunking → embedding → Qdrant upload.

### 6. Start the Chat Agent

```bash
python3 project2.py
```

```
=============================================
🧠 RAG Orchestrator Initialized (Type 'exit' to quit)
=============================================

You: 
```

Type `exit` or `quit` to shut down gracefully.

---

## ⚙️ How It Works — In Depth

### Semantic Cache (`semantic_cache.py`)

Before hitting the LLM, every user query is converted to a vector and compared against previously cached question-answer pairs in Qdrant. If the cosine similarity score of the best match exceeds the `similarity_threshold` (default: `0.80`), the cached answer is returned immediately — **zero LLM cost, zero latency**.

- Cache **hits** are logged with their similarity score.
- Cache **misses** are logged with the highest score that failed the threshold, so you can tune it over time.

### RAG Pipeline (`RAG.py`)

The document is chunked using a **sliding window** strategy (chunk size: 500 chars, overlap: 100 chars) to preserve context across chunk boundaries. Chunks under 150 characters and exact duplicates are filtered out before ingestion.

At query time, the top 5 most semantically similar chunks are retrieved from Qdrant and injected into the prompt as `Retrieved Context`.

Each chunk is assigned a **deterministic UUID** derived from an MD5 hash of its text content. This means re-ingesting the same document will `upsert` (not duplicate) vectors in Qdrant.

### Context Window Compression (`project2.py → _manage_context_window`)

The orchestrator maintains a rolling `conversation_history` list. When it exceeds `history_limit` (default: 6 messages), the older messages are sent to a dedicated **Summarizer Agent**, which compresses them into a dense system memory note. The 4 most recent messages are always preserved verbatim to maintain conversational flow.

This approach means the agent can handle long multi-turn conversations without ever hitting token limits.

### Observability with LogFire

Every meaningful operation is wrapped in a `logfire.span()`:

| Span | What it tracks |
|---|---|
| `orchestrator.chat_turn` | Full end-to-end turn latency |
| `orchestrator.compress_history` | Compression trigger, original vs. new history length |
| `qdrant_cache_search` | Cache hit/miss, similarity score |
| `qdrant_cache_save` | Saved point ID |
| `rag.retrieve_context` | Number of chunks retrieved |
| `rag.generate_embeddings` | Batch size, token cost |
| `rag.store_in_qdrant` | Number of points uploaded |
| `rag.full_ingestion_pipeline` | Full ingestion run |

---

## 🔧 Configuration Reference

| Parameter | Location | Default | Description |
|---|---|---|---|
| `history_limit` | `ConversationOrchestrator.__init__` | `6` | Max messages before compression triggers |
| `similarity_threshold` | `SemanticCache.__init__` | `0.80` | Minimum cosine score for a cache hit |
| `top_k` | `RAG.chunk_retrieval` | `5` | Number of RAG chunks to retrieve |
| `chunk_size` | `RAG.chunk_document` | `500` | Max characters per chunk |
| `overlap` | `RAG.chunk_document` | `100` | Overlap between adjacent chunks |
| `min_size` | `RAG.chunk_document` | `150` | Minimum chunk size (smaller are discarded) |
| `model` | `ConversationOrchestrator.__init__` | `gpt-4o-mini` | LLM used for both agents |
| `embedding_model` | Multiple | `text-embedding-3-small` | Embedding model (1536 dimensions) |

---

## 📁 Project Structure

```
.
├── project2.py        # Orchestrator + CLI entrypoint
├── RAG.py             # Ingestion pipeline + retrieval
├── semantic_cache.py  # Qdrant-backed semantic cache
├── rag_text.txt       # Source document (demo: Gift of the Magi)
├── requirements.txt
└── README.md
```

---

## 🛣️ Roadmap / Potential Improvements

- [ ] FastAPI wrapper to expose the agent as a REST endpoint
- [ ] Multi-document support with per-collection routing
- [ ] Streaming responses via `agent.iter()`
- [ ] Web UI (Streamlit or Gradio)
- [ ] Configurable embedding model and LLM via a `config.yaml`
- [ ] Automated cache TTL / expiry for stale answers

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the MIT License. The demo knowledge base (*The Gift of the Magi* by O. Henry) is in the public domain via [Project Gutenberg](https://www.gutenberg.org/ebooks/7256).

---

<p align="center">Built with ❤️ using PydanticAI · Qdrant · LogFire</p>
