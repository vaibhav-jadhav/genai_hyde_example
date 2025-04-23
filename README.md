# 🤖 HyDE-based RAG Retrieval Project

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using **HyDE (Hypothetical Document Embeddings)** to enhance retrieval results with OpenAI embeddings, Qdrant vector store, and LangChain components.

---

## 📁 Project Structure

```
.
├── .env                      # Contains your OpenAI API key
├── .gitignore               # Ignores .env, venv, and checkpoints
├── docker-compose.db.yml    # Docker setup for Qdrant vector database
├── hyde.py                  # Main Python script that performs RAG with HyDE
├── hyde_rag_example.ipynb   # Jupyter notebook version of the demo
├── python.pdf               # Input documentation file used for RAG
├── requirements.txt         # Python dependencies
├── venv/                    # Python virtual environment
└── .ipynb_checkpoints/      # Jupyter internal folder
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository and create a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 2. Add your OpenAI API key

Create a `.env` file in the root directory with the following content:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

---

### 3. Install all dependencies

```bash
pip install -r requirements.txt
```

You also need to install Jupyter Notebook if you want to use the notebook version:

```bash
pip install notebook
```

---

### 4. Start Qdrant Vector DB using Docker

Make sure Docker is installed and running. Then start Qdrant using:

```bash
docker-compose -f docker-compose.db.yml up
```

This will start Qdrant on `localhost:6333`

---

### 5. Run the script

You can either:
- Run the Python script:

```bash
python hyde.py
```

OR

- Launch the notebook version:

```bash
jupyter notebook hyde_rag_example.ipynb
```

---

## ✅ Output

The program will:
- Load the `python.pdf` documentation
- Embed and store it in Qdrant
- Run a query (with and without HyDE)
- Show relevant document chunks based on semantic similarity

---

## 📌 Notes

- Make sure `.env` is not committed to source control.
- You can change the PDF file, query, or model in the `hyde.py` script.
- This is a great base project to extend for your own RAG experiments or chatbot products.

---

Happy coding! 💻🚀
