##################################
# Author: Lefki Meidi            #
# MIT License                    #
# Copyright (c) 2025 Meidi Lefki #
##################################

import os
from sentence_transformers import SentenceTransformer
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import json 
import hashlib
'''
This file is for embedding purposes using pine, and fetching our chunks created
from chunky_cut to store em and insert them into pinecone

'''

JSONL_PATH = r"C:\Users\MLSD24\Desktop\chatbot\chunks_SQL_manual.jsonl"
load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-data")
NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "data_team_docs")

# let us fetch the chunks to retrieve content and metadatas

texts, metas = [] , []
with open(JSONL_PATH,"r",encoding="utf-8") as file:
    for line in file:
        c = json.loads(line)
        #text helps to fathom user's question
        texts.append(c["text"])
        #metas is utilized to quote the passage in the response after pine compared vectors
        metas.append({"doc":c["doc"],"page":c["page"],"text":c["text"][:400]})
    

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#the model is called upon, now it is time to vectorize and normalize our text, cosine is to be 1
#as we use cauchy-schwartz's inequality

vecs = model.encode(texts, normalize_embeddings=True, batch_size=64,show_progress_bar=True)


#check for pinecode index

api_key = os.getenv("PINECONE_API_KEY")
pine_index = os.getenv("PINECONE_INDEX")
assert api_key, "An error occured"

pinecone_connection = Pinecone(api_key=api_key)

exist = [i.name for i in pinecone_connection.list_indexes()]

if pine_index not in exist:
    pinecone_connection.create_index(name=pine_index,
                                     dimension=384,
                                     metric="cosine",
                                     spec=ServerlessSpec(cloud="aws",region="us-east-1")
                                     )

time.sleep(2)
connect = pinecone_connection.Index(pine_index)
print(connect)

#here we're building a hash, to prevent duplicates in the future
def make_id(m):
    s = f"{['doc']}::{m['page']}::{m['text']}"
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

#here v.list converts vectors previously made as pinecone doesnt accept
# pytorch/numpy vectors
records = [
    {"id": make_id(m), "values": v.tolist(), "metadata": m}
    for v, m in zip(vecs, metas)
]
print(f"[INFO] Prepared {len(records)} vectors")


#now we batch em up into 200 pieces each, then we upsert them

BATCH = 200
total = 0
for i in range(0, len(records), BATCH):
    chunk = records[i:i+BATCH]
    connect.upsert(vectors=chunk, namespace=NAMESPACE)
    total += len(chunk)
    print(f"[UPSERT] {total}/{len(records)}")
print(f"[DONE] Upserted {total} vectors into {INDEX_NAME} (ns={NAMESPACE})")

