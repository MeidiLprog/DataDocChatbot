##################################
# Author: Lefki Meidi            #
# MIT License                    #
# Copyright (c) 2025 Meidi Lefki #
##################################

'''
This file is made to establish a connection with huggingface
thus we shall use it to access/edit our data
'''

import os # access system's functions
import re #useful to retrieve str expressions using regular expression or functions such as findall
import itertools #useful to permute/group/aggregate data
from dotenv import load_dotenv #load env data here our api keys
from pypdf import PdfReader #read pdf files
from sentence_transformers import SentenceTransformer # allow us to create embedding vectors
from pinecone import Pinecone, ServerlessSpec   #Pinecode = db using vectors(useful for embeddings) | serverless = config serverless index
from huggingface_hub import login

print("Our first chatbot using LLM + RAG\n")

load_dotenv() # our env variables are loaded
tok_hf = os.getenv("HF_TOKEN")
assert tok_hf, "HuggingFace key successfully added !"
login(tok_hf)
print("Successfully connected to huggingface")


