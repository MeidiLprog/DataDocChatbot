# DataDocChatbot

DataDocChatbot is a **Retrieval-Augmented Generation (RAG)** assistant designed to answer questions from your private PDF documents related to SQL or data engineering, even when scanned (OCR).  
It extracts text, generates semantic embeddings, indexes them in Pinecone, and produces grounded answers using Groq (Llama-3.1) with verifiable citations.

---

## Overview

Complete RAG pipeline:

- PDF ingestion (text + scanned images)
- Text normalization & semantic chunking
- Embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Vector storage in Pinecone (cosine similarity)
- Retrieval of top relevant chunks
- Answer generation via Groq Llama-3.1-8B
- Web chatbot UI with Gradio (streaming)

**Why RAG?**  
LLMs do not know your private documents.  
RAG retrieves relevant text first and constrains the model to answer **only** from those excerpts → fewer hallucinations, traceable citations.

---

## Architecture

```
PDFs (docs/)
     |
[PyMuPDF + Tesseract OCR]
     |
     v
Normalize + Chunk  --->  chunks.jsonl
     |                       |
     v                       v
 [Sentence Transformers]   [Question embedding]
      |                           |
      v                           v
   Pinecone Upsert <--- Pinecone Query (top_k)
      |
      v
   RAG Prompt → Groq (Llama-3.1-8B) → Final Answer + Citations
```

---

## ✅ 3) Mathematics Used

> **Tip for GitHub READMEs:** use images for equations since native LaTeX ($$...$$) is not rendered.

### 📌 3.1 Sentence Embeddings

Each text chunk `t` becomes a 384‑dimensional semantic vector:  
![e(t) ∈ R^{384}](https://latex.codecogs.com/png.latex?e%28t%29%20%5Cin%20%5Cmathbb%7BR%7D%5E%7B384%7D)

All embeddings are L2‑normalized:  
![||e||_2 = 1](https://latex.codecogs.com/png.latex?%5ClVert%20e%20%5CrVert_2%20%3D%201%20%5Cquad%5CLongrightarrow%5Cquad%20%5Csum_%7Bi%3D1%7D%5E%7B384%7D%20e_i%5E2%20%3D%201)

This makes cosine similarity equal to the dot product.

---

### 📌 3.2 Cosine Similarity

Given question embedding `q` and chunk embedding `e`:  
![cos(theta) = frac{q ⋅ e}{||q|| ||e||}](https://latex.codecogs.com/png.latex?%5Ccos%28%5Ctheta%29%20%3D%20%5Cfrac%7Bq%20%5Ccdot%20e%7D%7B%5ClVert%20q%20%5CrVert%20%5ClVert%20e%20%5CrVert%7D)

With normalized vectors:  
![cos(theta) = q ⋅ e](https://latex.codecogs.com/png.latex?%5Ccos%28%5Ctheta%29%20%3D%20q%20%5Ccdot%20e)

Higher score ⇒ more relevant chunk. Pinecone ranks results with this metric.

---

### 📌 3.3 Overlapping Chunking

Splitting text into fixed‑size chunks can break sentences across boundaries.  
To preserve context, overlapping windows are used (e.g., 900 words, overlap 120).  
This increases recall and improves retrieval accuracy.

---

### 📌 3.4 Stable Vector IDs

Each vector uses a SHA‑1 hash of `(doc || page || text)`:  
![ID = SHA1(doc || page || text)](https://latex.codecogs.com/png.latex?ID%20%3D%20%5Cmathrm%7BSHA1%7D%28%5Ctext%7Bdoc%7D%20%7C%7C%20%5Ctext%7Bpage%7D%20%7C%7C%20%5Ctext%7Btext%7D%29)

This prevents duplicates and makes ingestion idempotent.


---

## Project Files

| File | Purpose |
|------|---------|
| `chunky_cut.py` | PDF → OCR → chunking → `.jsonl` |
| `embedding.py` | JSONL → embeddings → Pinecone upsert |
| `retrivial_question.py` | Retrieval debug (top‑k check) |
| `rag_answers.py` | Full RAG: retrieval + Groq + citations |
| `app.py` | Gradio chat UI (streaming) |

---

## Installation

```bash
python -m venv chatbot
source chatbot/Scripts/activate       # Windows: .\chatbot\Scripts\activate

pip install -U pip
pip install -U gradio==4.44.1 gradio_client
pip install sentence-transformers pinecone-client groq python-dotenv pymupdf pytesseract pillow
```

Create `.env`:

```
PINECONE_API_KEY=your_key
PINECONE_INDEX=rag-data
PINECONE_NAMESPACE=data_docs
GROQ_API_KEY=your_key
```

Windows OCR (if Tesseract not in PATH):

```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

## Running the Pipeline

### PDF → chunks.jsonl
```bash
python chunky_cut.py
```

### Embeddings → Pinecone
```bash
python embedding.py
```

### Test retrieval
```bash
python retrivial_question.py
```

### Launch chatbot
```bash
python app.py
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `TesseractNotFoundError` | Install Tesseract or set explicit path |
| Empty retrieval results | Check namespace + ingestion done |
| Gradio schema error | `pip install -U gradio==4.44.1` |
| Localhost blocked | launch with `server_name="127.0.0.1"` |

---

## Security

- Do **not** commit `.env` to Git
- Rotate API keys if exposed
- Separate Pinecone namespaces for dev/prod

---

## License

MIT — see `LICENSE`

---

## Acknowledgments

- Sentence Transformers  
- Pinecone  
- Groq (Llama‑3.1)  
- Gradio  
- PyMuPDF & Tesseract OCR

## Conclusion

A project carried out with dedication and commitment to unveil the secrets of RAGS and agents to acquire a solid foundation of the domain, and ready to bring it forth to a larger scale

![Gradio](images/first.gif)

![Gradio](images/second.gif)



