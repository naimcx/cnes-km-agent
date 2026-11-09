# CNES Knowledge Management Multi-Agent System (RAG)

A personal research and benchmarking project designed to evaluate and integrate modern GenAI orchestration frameworks (**LangChain**, **LlamaIndex**, and **LangGraph**). 

The system implements a production-grade multimodal Agentic RAG pipeline specialized in aerospace engineering documents, using official documents from Ariane or multiple tests documents.

---

## Project Objectives

1. **Hybrid & Multimodal Document Retrieval:**
   - Benchmark diverse vector retrieval architectures: LlamaIndex over Qdrant (dense vector + sparse BM25 hybrid search) vs. local LangChain backends (FAISS in-memory / ChromaDB).
   - High-precision engineering drawing parsing: Automated extraction of mechanical dimensions, diameter callouts ($\varnothing$), stage interfaces, and payload envelopes from PDF schematics via Vision-Language Models (VLMs).
2. **Actor-Critic Multi-Agent Workflow (LangGraph):**
   - Closed-loop orchestration with self-reflection and dynamic replanning.
   - Intelligent multi-source routing across internal aerospace manuals, external web tool (SearXNG), and Wikipedia.
   - Rigorous hallucination guardrails and aerospace metric verification powered by structured Pydantic schemas.

---

## Architecture & Workflow

The architecture relies on a cyclical Actor-Critic pattern governed by strict termination constraints:

![Architecture Graph](graph.png)

---

### Agent Roles

* **Orchestrator Node:** Analyzes query scope and assigns technical routing (`internal` manuals, `external` flight history / competitor launchers, or `hybrid` orbital mechanics comparison).
* **Researcher Node:** Executes search plans using internal hybrid vector search, SearXNG, Wikipedia, and an on-demand PDF image rendering and inspection tool.
* **Analyst Node:** Drafts comprehensive technical syntheses strictly grounded in retrieved evidence. Forbidden from extrapolating unverified values or confusing ground infrastructure with flight hardware.
* **Critic Node:** Evaluates the draft for factual precision and completeness, producing structured feedback (`is_faithful`, `answers_all_questions`).
* **Self-Reflection Node:** Performs root-cause analysis when feedback fails validation: initiates targeted `re_search` iterations or guides the analyst via `re_write` instructions.
* **Synthesizer Node:** Formats verified technical figures into Markdown tables and embeds certified schematic image artifacts.

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Agentic Framework & Graph** | LangGraph, LangChain |
| **RAG & Vector Stores** | LlamaIndex, Qdrant Client, ChromaDB, FAISS |
| **Embeddings** | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) |
| **LLMs & Inference** | Groq Cloud API (`openai/gpt-oss-120b`) or local Ollama (`llama3.2:3b` or `qwen2.5:7b`) |
| **Multimodal Vision** | PyMuPDF (native rendering), Llama-3.2-Vision / MiniCPM-V |
| **External Search Tools** | SearXNG self-hosted engine, Wikipedia-API |
| **Data Validation** | Pydantic schemas |

---

## Getting Started

### 1. Prerequisites

- Python 3.10+
- Running instance of **Qdrant** (port `6333`)
- (Optional) Running instance of **SearXNG** (port `8080`) and **Ollama** (port `11434`)

### 2. Environment Setup

Clone the repository and install the dependencies:

```bash
git clone [https://github.com/naimcx/cnes-km-agent.git](https://github.com/naimcx/cnes-km-agent.git)
cd cnes-km-agent
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create and configure your .env file:

```bash
cp .env.example .env
```

Ensure the key variables are configured:

```bash
TEXT_PROVIDER=groq # or ollama
VISION_PROVIDER=groq # or ollama
GROQ_API_KEY=your_groq_api_key

RAG_DB=qdrant
RAG_MODE=hybrid
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=cnes_km_agent

DATA_PATH=data
IMG_PATH=static/pdf_images
```

Start the required infrastructure services (Qdrant, SearXNG) in Docker

```bash
docker compose up -d
```

### 3. Data Ingestion Pipeline

Place the technical document (ex: Mua-6_Issue-2_Revision-0_March-2021.pdf) inside the data/ directory, then launch the incremental ingestion pipeline:

```bash
python src/ingest_pipeline.py
```

### 4. CLI Execution

Run queries directly through the multi-agent graph:

```bash
# Interactive mode
python main.py

# Direct question execution
python main.py --db qdrant --mode hybrid --question "What is the usable volume within the A62 for a fairing in a single launch? I need the diameter at the very top and at the base."
```
