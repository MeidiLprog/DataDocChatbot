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
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-data")
NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "data_team_docs")
API_KEY    = os.environ["PINECONE_API_KEY"]

#question encoding

question = "How do I do a select in SQL ?"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qvec = model.encode([question],normalize_embeddings=True)[0].tolist()


#query pinecone
connect = Pinecone(api_key=API_KEY)
index = connect.Index(INDEX_NAME)
res = index.query(vector=qvec,top_k=6,include_metadata=True,namespace=NAMESPACE)

for i, m in enumerate(res.matches, 1):
    meta = m["metadata"]
    print(f"[{i}] score={m['score']:.3f}  {meta['doc']} p.{meta['page']}")
    print(meta["text"])
    print("-"*80)