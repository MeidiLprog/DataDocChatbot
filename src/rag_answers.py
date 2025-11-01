##################################
# Author: Lefki Meidi            #
# MIT License                    #
# Copyright (c) 2025 Meidi Lefki #
##################################

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from groq import Groq  # <--- NEW

load_dotenv()
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-data")
NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "data_team_docs")
API_KEY    = os.environ["PINECONE_API_KEY"]
GROQ_KEY   = os.environ["GROQ_API_KEY"]  # doit être défini dans .env

# -------- prompt builder (RAG strict) --------
def build_prompt(question, matches):
    ctx = "\n\n".join([f"[{i+1}] {m['metadata']['text']}" for i, m in enumerate(matches)])
    return (
        "You are a concise SQL/Pandas/ML assistant. Answer ONLY using the provided excerpts and cite them [1],[2],...\n"
        "If the answer is not in the excerpts, say you don't know.\n"
        "Answer in the user's language (French or English) based on the question.\n\n"
        f"QUESTION: {question}\n\nEXCERPTS:\n{ctx}\n\nANSWER:"
    )

def ask(question: str):
    # We embed the question ans convert the first vector from a numpy vector to a python list
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qv = model.encode([question], normalize_embeddings=True)[0].tolist()

    # Retrieval Pinecone
    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(INDEX_NAME)
    res = index.query(vector=qv, top_k=6, include_metadata=True, namespace=NAMESPACE)
    matches = res.matches or []

    # If doesn't understand the question it says it found nothing relevant
    if not matches:
        return "Sorry.. I have found no result answering your querying, please ask another question.\n"

    # We the build prompt
    prompt = build_prompt(question, matches)

    # Call upon Groq to reply
    gclient = Groq(api_key=GROQ_KEY)
    chat = gclient.chat.completions.create(
        model="llama-3.1-8b-instant",  # Free and fast model
        messages=[
            {"role": "system",
             "content": "You are a careful RAG assistant. Only use the provided excerpts and cite them [1],[2]... If info is missing, say you don't know."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=350,
    )
    answer = chat.choices[0].message.content

    # Call upon our metadata
    cites = "\n".join(
        [f"[{i+1}] {m['metadata']['doc']} p.{m['metadata']['page']}" for i, m in enumerate(matches)]
    )
    return f"{answer}\n\n{cites}"

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How do I do a SELECT in SQL?"
    print(ask(q))
