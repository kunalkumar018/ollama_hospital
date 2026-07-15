# 🏥 Ollama Hospital RAG Demo

A simple **Retrieval-Augmented Generation (RAG)** application built with **Streamlit** and **Ollama**.  
It answers questions about hospitals using a curated dataset of hospital facts.  
The app embeds a knowledge base, retrieves the most relevant facts for each query, and generates a contextual response using a lightweight language model.

---

## ✨ Features

- **Local RAG** – all processing runs offline via Ollama.
- **Vector database** – embeddings are computed once and cached for fast retrieval.
- **Dynamic context display** – the sidebar shows which facts were used to answer your question.
- **Streaming responses** – the model's answer appears token by token.
- **Fallback dataset** – if `Hospital-facts.txt` is missing, the app creates a minimal one automatically.

---

## 📦 Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally
- The following Python packages (all listed in `requirements.txt`):

```txt
streamlit
pandas
numpy
matplotlib
ollama
```

*Note:* `pandas`, `numpy`, and `matplotlib` are not directly used but are kept for potential extensions.

---

## 🚀 Installation & Setup

### 1. Clone or download the project files

Make sure you have the following files in your project directory:

- `streamlit-app.py`
- `Hospital-facts.txt` (optional; the app will create a fallback if missing)
- `requirements.txt`

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and run Ollama

Follow the official [Ollama installation guide](https://ollama.com/download).  
Once installed, ensure the Ollama service is running:

```bash
ollama serve
```

### 4. Pull the required models

The app uses two models from Hugging Face (GGUF versions):

- **Embedding model**: `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf`  
- **Language model**: `hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF`

Pull them with:

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

> ⚠️ These models are relatively small and should run on most machines.  
> You can replace them with any Ollama-compatible models by editing the `EMBEDDING_MODEL` and `LANGUAGE_MODEL` variables in `streamlit-app.py`.

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run streamlit-app.py
```

The app will open in your browser.

- **Ask a question** about hospitals in the text box (e.g., *"What is the oldest hospital?"*).
- The **sidebar** will show the top‑3 most relevant facts from the dataset, along with similarity scores.
- The **main area** will display the model’s answer, streamed in real time.

---

## 🧠 How It Works

1. **Initialization** – the app reads `Hospital-facts.txt`, splits it into chunks (one fact per line), and computes an embedding for each chunk using the embedding model. These are stored in memory as a simple vector database.

2. **Retrieval** – when you submit a query, the app:
   - Computes the query embedding.
   - Compares it to all chunk embeddings using **cosine similarity**.
   - Returns the top‑3 most similar facts.

3. **Generation** – the retrieved facts are injected into a system prompt. The language model then generates a final answer, using *only* the provided context.

4. **UI** – the sidebar shows the retrieved chunks with their scores, and the response is streamed token by token.

---

## ⚙️ Configuration

You can adjust the following in `streamlit-app.py`:

- `EMBEDDING_MODEL` – change the model used for embeddings.
- `LANGUAGE_MODEL` – change the generative model.
- `top_n` parameter in the `retrieve()` function – controls how many facts are retrieved (default is 3).
- `dataset_path` – the name of your facts file.

---

## 📁 Dataset

The app uses `Hospital-facts.txt`, which contains over 100 interesting facts about hospitals.  
If the file is missing, the app creates a minimal fallback file on first run to avoid crashing.

---

## 🐛 Troubleshooting

- **Ollama connection errors** – make sure Ollama is running (`ollama serve`) and the models are pulled correctly.
- **Slow first start** – the first run computes embeddings for every fact; subsequent runs use the cached vector DB (thanks to `@st.cache_resource`).
- **Out‑of‑memory issues** – try using smaller models or reduce the dataset size.
- **Model not found** – double‑check the model names; you can list installed models with `ollama list`.


## 🙌 Acknowledgements

- [Ollama](https://ollama.com/) for the local model runtime.
- [Streamlit](https://streamlit.io/) for the interactive UI framework.
- Hugging Face for the model weights.
