# CNES Knowledge Management Multi-Agent System (RAG)

A personal research and benchmarking project designed to evaluate and integrate modern GenAI orchestration frameworks (**LangChain**, **LlamaIndex**, and **LangGraph**).

The system combines **LangChain** and **LangGraph** for modular tool calling, state management, and cyclic multi-agent orchestration, **LlamaIndex** with **Qdrant** for advanced document ingestion and hybrid retrieval (dense embeddings + sparse BM25), specialized **Skills** for CAD schematic inspection, and a sandboxed **Docker** environment for deterministic aerospace math calculations.

---

## Project Objectives

1. **Hybrid & Multimodal Document Retrieval:**
   - Benchmark diverse vector retrieval architectures: LlamaIndex over Qdrant (dense vector + sparse BM25 hybrid search) vs. local LangChain backends (FAISS in-memory / ChromaDB).
   - High-precision engineering drawing parsing: Automated extraction of mechanical dimensions, diameter callouts ($\varnothing$), stage interfaces, and payload envelopes from PDF schematics via Vision-Language Models (VLMs).
2. **Actor-Critic Multi-Agent Workflow (LangGraph):**
   - Closed-loop orchestration with autonomous self-reflection and dynamic replanning.
   - Intelligent multi-source routing across internal aerospace manuals, an on-demand PDF visual inspector, external web tools (SearXNG), and Wikipedia.
   - Rigorous hallucination guardrails powered by structured Pydantic schemas.
3. **Deterministic Math & Sandbox Execution:**
   - Isolated code execution in a Docker sandbox for physical calculations (Tsiolkovsky delta-V, geometric fairing clearances) to eliminate LLM arithmetic hallucinations.
4. **Air-Gapped Sovereign Deployment:**
   - Full compatibility with local hardware via Ollama (Qwen 2.5, MiniCPM-V) for zero-data-leakage deployment in secure aerospace environments.

---

## Architecture & System Workflow

The architecture follows a deterministic Actor-Critic cycle with autonomous self-reflection and dynamic replanning:

```text
[ User Query ]
       │
       ▼
[ Orchestrator ] ──(Scope: internal / external / hybrid)──► [ Researcher ] ◄────────┐
                                                                 │                  │
                                                                 ▼             (re_search)
                                                            [ Analyst ] ◄─────┐     │
                                                                 │            │     │
                                                                 ▼            │     │
                                                             [ Critic ]  (re_write) │
                                                            /          \      │     │
                                          (is_valid == True)         (False)  │     │
                                                   /                      \   │     │
                                                  ▼                        ▼  │     │
                                           [ Synthesizer ]        [ Self-Reflection ]
                                                  │
                                                  ▼
                                       [ Validated Deliverable ]
```

![Architecture Graph](graph.png)

---

### Agent Roles & Infrastructure Services

- **Orchestrator (`prompts/orchestrator.md`):** Classifies the inquiry domain (`internal`, `external`, `hybrid`) and formats keyword-dense search plans.
- **Researcher (`prompts/researcher.md`):** Executes documentary retrieval, follows document figure cross-references (page $N \to N+1$), inspects CAD diagrams, and invokes the Docker calculation sandbox.
- **Analyst (`prompts/analyst.md`):** Synthesizes technical metrics, handles multi-variant configuration matrices (e.g., Short vs. Long fairings), and enforces strict zero-hallucination rules.
- **Critic (`prompts/critic.md`):** Quality assurance auditor verifying literal context grounding, absence of assumed numbers, and separation between ground and flight systems (`is_valid = True/False`).
- **Self-Reflection (`prompts/self_reflection.md`):** Root-cause evaluator triggered upon QA rejection. Determines whether to route to `researcher` (`re_search` with refined queries) or directly back to `analyst` (`re_write` instructions).
- **Synthesizer (`prompts/synthesizer.md`):** Normalizes dimensions into Markdown tables and attaches verified drawing assets.
- **CAD & Math Skills (`skills/`):** Dedicated system instructions for technical drawing visual extraction (`cad-inspector`) and formula verification (`aerospace-calc`).

---

## Tech Stack

| Component | Technologies |
| :--- | :--- |
| **Agentic Framework** | LangGraph, LangChain Core |
| **RAG & Hybrid Search** | LlamaIndex, Qdrant (Dense + BM25), ChromaDB, FAISS |
| **PDF Extraction Engine** | PyMuPDF (Fitz), IBM Docling, Unstructured |
| **Embeddings** | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) |
| **LLM Inference** | **Cloud:** Groq API (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`)<br>**Local:** Ollama (`qwen2.5:7b`, `llama3.2:3b`) |
| **Vision Models (VLM)** | **Cloud:** Groq Vision (`qwen/qwen3.8-27b`)<br>**Local:** Ollama (`minicpm-v`, `qwen2.5-vl:7b`) |
| **Execution Sandboxing** | Docker (Isolated Python environment with NumPy & SciPy) |
| **Local Search Engine** | SearXNG (Self-hosted container) |
| **Data Validation** | Pydantic schemas |

---

## Getting Started

### 1. Prerequisites

- **Python 3.10+**
- **Docker** (running for **Qdrant**, **SearXNG** and the **calculation sandbox**)
- (Optional for local inference) **Ollama** installed with models pulled :
```bash
ollama pull qwen2.5:7b
ollama pull minicpm-v
```

### 2. Environment Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/naimcx/cnes-km-agent.git
cd cnes-km-agent
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Configuration

Create and configure your .env file:

```bash
cp .env.example .env
```

Configure your target execution mode in `.env`:

```ini
# --- PROVIDER CONFIGURATION (groq OR ollama) ---
TEXT_PROVIDER=groq
VISION_PROVIDER=groq

# Cloud API Keys
GROQ_API_KEY=your_groq_api_key

# Model Selection
GROQ_TEXT_MODEL=openai/gpt-oss-120b
GROQ_VISION_MODEL=qwen/qwen3.8-27b
LOCAL_TEXT_MODEL=qwen2.5:7b
LOCAL_VISION_MODEL=minicpm-v

# --- RETRIEVAL & VECTOR DATABASE ---
RAG_DB=qdrant
RAG_MODE=hybrid
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=cnes_km_agent
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# --- DOCUMENT & PARSER CONFIGURATION ---
PDF_PARSER_TYPE=pymupdf       # Options: pymupdf, docling, unstructured
DATA_PATH=data
IMG_PATH=static/pdf_images
PRE_RENDER_IMAGES=true
```

### 4. Start Infrastructure Containers

Start Qdrant and SearXNG via Docker Compose

```bash
docker compose up -d
```

Verify services are healthy:
- Qdrant Web UI: `http://localhost:6333/dashboard`
- SearXNG: `http://localhost:8080`

---

## Ingestion & Document Indexing

Place the technical document (ex: Mua-6_Issue-2_Revision-0_March-2021.pdf) inside the data/ directory, then launch the incremental ingestion pipeline:

```bash
# Incremental indexing (skips already indexed files using SHA-256)
python -m src.ingest_pipeline

# Force re-indexing of all documents
python -m src.ingest_pipeline --force
```

---

## CLI Execution

`main.py` provides an interactive shell and supports non-destructive CLI parameter overrides:

### Interactive Execution
```bash
python main.py
```

### Direct Headless Execution
```bash
python main.py --question "What are the separation systems on the Ariane 6 fairing?"
```

### Running 100% Locally with Ollama
```bash
python main.py --provider ollama --text-model qwen2.5:7b --vision-model minicpm-v --question "What is the usable volume within the A62 for a fairing in a single launch?"
```

### Testing Different Vector Stores & Modes
```bash
# Vector search only via Qdrant
python main.py --db qdrant --mode vector

# In-memory FAISS fallback
python main.py --db faiss --mode keyword
```

---

## Verification & Testing

Execute the test suites via `pytest`:

```bash
# Run all unit and integration tests
pytest

# Test tools, skills, prompt loading, and Docker math execution
pytest tests/test_tools_and_skills.py -v -s

# Run parser extraction benchmarks (PyMuPDF vs Docling vs Unstructured)
python tests/test_benchmark_parsers.py
```

---

## Technical Roadmap & Future Enhancements

Planned architectural evolutions and deferred features for production scaling:

- **GraphRAG & Knowledge Graphs:** Extract aerospace entities and ontologies (e.g., Neo4j / RDF) to link mechanical requirements, materials, and flight telemetry.
- **Model Context Protocol (MCP):** Expose tools and databases (SearXNG, Qdrant, Mission SQL) as standardized MCP servers via `langchain-mcp-adapters`.
- **ColPali & Late-Interaction Retrieval:** Multi-vector page-level indexing with Binary Quantization in Qdrant to improve visual and tabular recall without loss of context.
- **Human-in-the-Loop Conversational Mode:** Dynamic interruption (`interrupt`) in LangGraph when the Orchestrator detects ambiguous technical requirements.
- **Multi-Document Targeted Inspection:** Cross-reference CAD schematics across multiple distinct PDF manuals and mission revisions (`inspect_pdf_page_image_v2`).
- **Structured Observability:** Standardized `logging` infrastructure with controllable `LOG_LEVEL` across nodes and sandbox executions.
- **Modern Web Interface:** Interactive multimodal dashboard for real-time trace inspection and graph streaming with NextJS
