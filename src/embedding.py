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
'''
This file is for embedding purposes using pine, and fetching our chunks created
from chunky_cut to store em and insert them into pinecone

'''

JSONL_PATH = r"...\chunks_SQL_manual.jsonl"
INDEX_NAME = os.getenv("PINECONE_INDEX", "rag-data")
NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "data_team_docs")

# let us fetch the chunks to retrieve content and metadatas

text, metas = [] , []
with open(JSONL_PATH,"r",encoding="utf-8") as file:
    for line in file:
        c = json.loads(line)
        #text helps to fathom user's question
        text.append(c["text"])
        #metas is utilized to quote the passage in the response after pine compared vectors
        metas.append({"doc":c["doc"],"page":c["page"],"text":c["text"][:400]})
    

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#the model is called upon, now it is time to vectorize and normalize our text, cosine is to be 1
#as we use cauchy-schwartz's inequality

vecs = model.encode(text, normalize_embedding=True, batch_size=64,show_progress_bar=True)


#check for pinecode index

load_dotenv()
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

time.sleep(5)
connect = pinecone_connection.Index(pine_index)
print(connect)