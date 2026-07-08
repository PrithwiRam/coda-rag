# CODA-RAG: Constraint-Oriented Dual-Agent RAG

**A Resource-Constrained Agentic Retrieval-Curation Framework**

CODA-RAG is a research-grade codebase designed to explore and benchmark constraint-compliant multi-agent retrieval architectures. By leveraging Large Language Model (LLM) orchestration in a stateful reasoning loop, it dynamically routes queries between different specialized retrieval agents under strict latency and token budgets.

---

## 🚀 Key Features

*   **Stateful ReAct Coordinator**: Dynamically schedules query actions and sub-queries using a reasoning-action-observation scratchpad loop.
*   **Dual-Agent Execution**: 
    *   *Exploitation Agent*: Highly efficient Lexical retrieval (BM25) for entity-heavy and factual lookups.
    *   *Exploration Agent*: Dense vector semantic search (FAISS) for broad conceptual understanding.
*   **Constraint Enforcement Gate (MOSAIC)**: Formulates context assembly as a submodular optimization problem, dynamically relaxing objective weights to fit tight context token limits.
*   **Multi-Objective Optimization**: Evaluates relevance, latency, redundancy, and confidence to adaptively guide planner routing.

---

## 📦 Installation

To set up the environment, run the following:

```bash
# Clone the repository
git clone https://github.com/PrithwiRam/coda-rag.git
cd coda-rag

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Usage

### Running the Demo
To execute the pipeline and view the ReAct coordinator's reasoning traces and scratchpad steps:

```bash
python main.py
```

### Running the Benchmark
To evaluate the system against baseline RAG pipelines (lexical, dense, hybrid):

```bash
python run_evaluation.py
```
