##################################
# Author: Lefki Meidi            #
# MIT License                    #
# Copyright (c) 2025 Meidi Lefki #
##################################

'''
This file is used to create or use our index from pinecone
it shall later be used to store our vectors
'''

import os
from dotenv import load_dotenv
import re
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone,ServerlessSpec
import itertools
import time

print("Pinecone Set up \n")

load_dotenv() # as usual, we load our variables 
api_key = os.getenv("PINECONE_API_KEY")
assert api_key, "pinecone key succesfully loaded !"
index_pin = os.getenv("PINECONE_INDEX")
assert index_pin, "Index loaded !\n"

print("PINECONE SUCCESSFULLY LOADED ! \n")

#connection to pinecone
pinecone_connection = Pinecone(api_key=api_key)

#if index not created let's set up one
# dim of sentence_transformers = 384
dim = 384
#iterate over indexes available and store it in existing
existing = [i.name for i in pinecone_connection.list_indexes()]

#check whether index_pin exist aka rag-data
#if not we define it
if index_pin not in existing:
    print(f"Index {index_pin} doesn't exist\n")
    
    # for metric here we use cosine, implying cauchy-schwartz's inequality to measure our cos between 2 vectors
    pinecone_connection.create_index(
                                     name=index_pin,
                                     dimension=dim,
                                     metric="cosine",
                                     spec=ServerlessSpec(cloud="aws",region="us-east-1"))
else:
    print("Index already defined ! \n")

time.sleep(5)
try: 
    index = pinecone_connection.Index(index_pin)
    assert index, "Connection established !"
except Exception as e:
    print("An error occured while trying to establish the connection \n")
    raise RuntimeError("Index not ready ! \n")

