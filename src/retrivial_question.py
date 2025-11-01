##################################
# Author: Lefki Meidi            #
# MIT License                    #
# Copyright (c) 2025 Meidi Lefki #
##################################

'''
This file is a test, since embedding is done we now try to query and retrieve
answers to check whether our whole pipeline works before adding huggingface

'''


import os
from sentence_transformers import SentenceTransformer
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import json 


load_dotenv()
pine_index = os.getenv("PINECONE_INDEX", "rag-data")
NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "data_team_docs")
API_KEY    = os.environ["PINECONE_API_KEY"]

#question encoding

question = "How do I do a select in SQL ?"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qvec = model.encode([question],normalize_embeddings=True)[0].tolist()


#query pinecone
connect = Pinecone(api_key=API_KEY)
time.sleep(2)
index = connect.Index(pine_index)

#debugging as I had difficulties with my vectors not forming nor detected
stats = index.describe_index_stats()
print("[STATS]", stats)

# display the namespace, here I seek "data_teams_docs"
total_vecs = stats.get("total_vector_count", 0)
namespaces = stats.get("namespaces", {})
print(f"[STATS] total_vector_count={total_vecs}")
print(f"[STATS] namespaces={list(namespaces.keys())}")

# In case the namespace is empty or non existing:
ns_count = namespaces.get(NAMESPACE, {}).get("vector_count", 0)
print(f"[STATS] vector_count in '{NAMESPACE}' = {ns_count}")



res = index.query(vector=qvec,top_k=6,include_metadata=True,namespace=NAMESPACE)

# "accuracy" or rather cos comparaison -1 <= cos <= 1 between the question and the text
for i, m in enumerate(res.matches, 1):
    meta = m["metadata"]
    print(f"[{i}] score={m['score']:.2f}  {meta['doc']} p.{meta['page']}")
    print(meta["text"])
    print("-"*80)