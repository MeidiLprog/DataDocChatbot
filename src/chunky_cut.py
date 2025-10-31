##################################
# Author: Lefki Meidi            #
# MIT License                    #
# Copyright (c) 2025 Meidi Lefki #
##################################
'''
This file utilizes pdf reader, and various functions
to chop up and break down our pdf documents into chunks
that we shall use later on and vectorize em up to store them
in pinecone

'''

import os
import re
from pypdf import PdfReader
from collections import defaultdict #useful to prevent KeyErrors to be raised

def info_pdf(func):
    def wrapper(*args,**kwargs):
        print("This function is merely useful just to display the first 300 caracters of a pdf file\n")
        return func(*args,**kwargs)
    return wrapper


#Set up part

DATA_DIR = r"C:\Users\MLSD24\Desktop\chatbot\docs"

CHUNK_WORDS = 900
OVERLAP = 120
MIN_CHARS = 200
SHOW_EXAMPLES_PER_DOC = 2

#End set up

@info_pdf
def test_read(path_file : str) -> None:

    reader = PdfReader(path_file)
    page = reader.pages[0]
    trc = page.extract_text()
    print(trc[:300])

    return

#now that our test worked out fine
'''
Let us build functions to break down our text
first one is about normalizing a page
so we remove excedants of spaces and returns
so we have something exploitable
'''
#we compile in advance to make the search easier and quicker
NORMALIZE_SPACES = re.compile(r"\s+")

def normalize(text : str) -> str:

    text = text or "" #we switch empty NONES in empty
    return NORMALIZE_SPACES.sub(" ",text).strip()

#now we can break down our text into chunks

def file_exist(NAME_OF_FILE: str):
    return True if os.path.isfile(NAME_OF_FILE) else False


print("Chopping our documents \n")

def chop_chunks(text : str, max_words : int ,overlap : int) -> list:
    words = text.split() #I got a list now that I can easily work with
    chunks = []
    i : int = 0
    while i < len(words):
        piece = " ".join(words[i:(i+max_words)]).strip() # we add the sentence from index i to i + maxwords and strip away spaces
        if piece:
            chunks.append(piece)
        i += max_words - overlap # here we substract the overlap to not stamp on words
    return chunks


def chunk_pdf(PDF_PATH : str, 
              MAX_WORDS : int, 
              OVERLAP : int, 
              MINCHAR = MIN_CHARS) -> list[dict]:
    if file_exist(PDF_PATH) == True:
        print("File verified !")

        reader = PdfReader(PDF_PATH)
        docname = os.path.basename(PDF_PATH)
        STORE_DATA = [] #array of dictionnaries
    #for a pdf file, we get the index of the page and the page
    # then we clean it(normalize) then we cut in chunks and append it to a dictionnary
        for index_page, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text()
            clean = normalize(raw_text)
            if len(clean) < 200:
                continue
        #here I retrieve the chunks and for each chunks I store it in a dictinnnary
            for ch in chop_chunks(clean,MAX_WORDS,OVERLAP):
                if len(ch) >= 200:
                    STORE_DATA.append(
                        {
                            "doc" : docname,
                            "page" : index_page,
                            "text" : ch
                            }
                            )

    return STORE_DATA


print("Let us check whether our directories exist !\n")


if __name__ == "__main__":
    assert os.path.isdir(DATA_DIR)
    #list the number of pdf files we got in our directory
    print(f"File's content: {os.listdir(DATA_DIR)}",sep="\n",end="\n")
    PDF_F = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf") ]
    assert PDF_F, "no pdf !"

    test_read(r"C:\Users\MLSD24\Desktop\chatbot\docs\pandas.pdf")


