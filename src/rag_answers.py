##################################
# Author: Lefki Meidi            #
# MIT License                    #
# Copyright (c) 2025 Meidi Lefki #
##################################



import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()
pine_index = os.getenv("PINECONE_INDEX", "rag-data")
NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "data_team_docs")
API_KEY    = os.environ["PINECONE_API_KEY"]

# using tiny llama for locals as errors occured while using hugginface
HF_MODEL_LOCAL = os.getenv("HF_MODEL_LOCAL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

#we load the model at lunch
print(f"[LOAD] Local model: {HF_MODEL_LOCAL}")
_local_tok = AutoTokenizer.from_pretrained(HF_MODEL_LOCAL)
_local_mdl = AutoModelForCausalLM.from_pretrained(HF_MODEL_LOCAL)
_local_pipe = pipeline("text-generation", model=_local_mdl, tokenizer=_local_tok)

def build_prompt(question, matches):
    ctx = "\n\n".join([f"[{i+1}] {m['metadata']['text']}" for i, m in enumerate(matches)])
    # simple prompt
    return (
        "You are a concise SQL/Pandas/ML assistant. Answer ONLY using the provided excerpts and cite them [1], [2], ...\n"
        "If the answer is not in the excerpts, say you don't know.\n"
        "Answer in the user's language (English) based on the question.\n\n"
        f"QUESTION: {question}\n\nEXCERPTS:\n{ctx}\n\nANSWER:"
    )

def ask(question: str):
    # First juste like inretrivial question I embeded my questio creating a numpy vector that I transform in list
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    qv = model.encode([question], normalize_embeddings=True)[0].tolist()

    # retrieve
    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(pine_index)
    res = index.query(vector=qv, top_k=6, include_metadata=True, namespace=NAMESPACE)
    matches = res.matches

    # prompt + local generation
    prompt = build_prompt(question, matches)
    out = _local_pipe(prompt, max_new_tokens=300, do_sample=False, temperature=0.2)
    answer = out[0]["generated_text"]

    # Quote our source with our meta data, doc,page, text
    cites = "\n".join([f"[{i+1}] {m['metadata']['doc']} p.{m['metadata']['page']}" for i, m in enumerate(matches)])
    return answer + "\n\n" + cites

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:])
    print(ask(q))


