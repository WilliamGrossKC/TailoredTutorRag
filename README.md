
# TailoredTutorRag

**TailoredTutorRag** is a flexible and modular **Retrieval-Augmented Generation (RAG)** pipeline built to enhance the performance of AI tutoring agents—specifically designed for [Tailored Tutor](https://tailoredtutor.org) to inject contextual knowledge from syllabi and textbooks into LLM prompts.

This project also includes a benchmarking framework to determine the optimal RAG configuration by testing different retrieval algorithms (e.g., BM25, TF-IDF), chunk sizes, and embedding strategies.

---

## 🚀 Features

- 🔍 **Contextual Document Injection**  
  Automatically pads LLM prompts with relevant context extracted from syllabi, textbooks, or academic resources.

- ⚙️ **RAG Configuration Benchmarking**  
  Test and evaluate combinations of:
  - Chunk sizes (8 to 256 tokens)
  - Retrieval algorithms (e.g., TF-IDF, BM25)
  - Embedding windows and stride
  - Preprocessing techniques

- 📊 **Performance Metrics**  
  Outputs accuracy scores for each config, enabling fine-grained performance comparisons across multiple RAG strategies.

- 🧠 **AI Tutoring Compatibility**  
  Designed to work as a backend wrapper for AI tutoring agents like Tailored Tutor.

---

## 📁 Directory Overview

- `llm_connection.py`  
  Handles API interactions with the LLM and manages prompt assembly.

- `syllabus_chunking.py`  
  Processes and chunks syllabus content into vectorizable units for retrieval.

- `utils.py`  
  Shared utility functions for formatting, loading, and evaluation.

- `rag_implementation.py`  
  Implements the actual RAG logic, including algorithm switching and evaluation harness.

- `algorithm_results.txt`  
  Contains experiment results from various configurations (e.g., chunk size, retrieval method), showing accuracy trends.

---

## 📈 Example Results

From `algorithm_results.txt`, we can see chunk size and stride significantly impact RAG performance. For instance:

- **Chunk Size 64, Stride 8**:  
  Scores consistently near **95-100%**, outperforming lower chunk sizes with narrow stride.

- **Chunk Size 8, Stride 1**:  
  Shows more variable performance and sometimes drops to **67-75%**.

This implies that **larger chunks and wider strides** offer better performance in this tutoring context.

---

## 🛠️ Installation

```bash
git clone https://github.com/WilliamGrossKC/TailoredTutorRag.git
cd TailoredTutorRag
pip install -r requirements.txt
```

> You may need to install `faiss`, `nltk`, or `sklearn` separately depending on your platform.

---

## 🧪 Usage

Run experiments with different RAG settings:

```bash
python rag_implementation.py --chunk_size 64 --stride 8 --algorithm bm25
```

Or simply run the wrapper for inference:

```bash
python llm_connection.py --input "What topics are covered in week 3 of my course?"
```

---

## 📚 Example Use Case

**Prompt:**  
> “What topics will be covered in the second exam?”

**With TailoredTutorRag**, the system searches through the uploaded syllabus and textbook, finds all mentions of Exam 2 content, and pads the prompt for the LLM with relevant sections.  
This greatly improves response accuracy and contextual awareness.

---

## 🧠 Future Improvements

- Add UI for configuring experiments interactively  
- Expand retrieval support to dense vector stores (e.g., FAISS, Qdrant)  
- Integrate textbook PDF parsing  
- Multi-document support and chunk overlap experimentation

---

## 📜 License

MIT License
