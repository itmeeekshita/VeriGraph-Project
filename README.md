# 🕸️ VeriGraph AI: Mitigating Hallucinations in LLMs using Graph-Based Consensus

**VeriGraph AI** is a hallucination mitigation system that improves the reliability of Large Language Models (LLMs) by replacing simple "majority voting" with a **semantic graph consensus mechanism**.

Instead of relying on a single AI response, VeriGraph orchestrates a debate between three distinct AI agents (Standard, Reasoning, and Critic). It maps their answers as nodes in a graph, calculates semantic similarity, and uses **Degree Centrality** to mathematically identify the most logical and verified answer.

---

##  Key Features

* **Multi-Agent Architecture:** Generates three distinct perspectives (Standard Response, Chain-of-Thought Reasoning, Adversarial Critique) in parallel using **Groq (Llama-3)**.
* **Graph-Based Verification:** Constructs a semantic graph where edges represent cosine similarity between answers.
* **Hallucination Filtering:** Uses **NetworkX** to compute centrality scores, automatically filtering out "outlier" hallucinations that lack logical support.
* **RAG Integration:** Supports PDF uploads to ground answers in specific documents (Retrieval-Augmented Generation).
* **"Trap" Detection:** Includes a demo mode capable of exposing common LLM failures (e.g., fake academic papers, historical impossibilities).

---

##  System Architecture

The system operates on a **5-Layer Pipeline**:

1.  **Input Layer:** Captures the user query (web interface).
2.  **Multi-Agent Layer:**
    * *Standard Agent:* Generates a direct answer.
    * *Reasoning Agent:* Uses Chain-of-Thought (CoT) to verify facts step-by-step.
    * *Critic Agent:* Actively tries to find flaws or hallucinations in the standard answer.
3.  **Vectorization Layer:** Converts text outputs into high-dimensional semantic embeddings using **HuggingFace (`all-MiniLM-L6-v2`)**.
4.  **Graph Logic Layer:**
    * Constructs an adjacency matrix based on **Cosine Similarity**.
    * Computes **Degree Centrality** to determine the "winning" node.
5.  **Output Layer:** Selects and displays the response with the highest centrality score as the verified truth.

---

##  Tech Stack

* **Frontend:** Streamlit
* **LLM Engine:** Groq API (Llama-3-70b / Llama-3-8b)
* **Orchestration:** LangChain
* **Vector Database:** FAISS (for RAG)
* **Embeddings:** HuggingFace / SentenceTransformers
* **Graph Computation:** NetworkX, Scikit-learn
* **PDF Processing:** PyPDF2

---

## Installation & Setup

Follow these steps to set up the project locally on your machine.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/itmeeekshita/VeriGraph-Project.git](https://github.com/itmeeekshita/VeriGraph-Project.git)
    cd VeriGraph-Project
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

5.  **Configuration**
    * The app will launch in your default web browser (usually at `http://localhost:8501`).
    * You will need a **Groq API Key** to run the agents. You can get one for free at [console.groq.com](https://console.groq.com).
