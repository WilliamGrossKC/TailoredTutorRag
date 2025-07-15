import os
import docx
import PyPDF2
import pdfplumber
import pypandoc
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import time
import nltk
import json
from nltk.tokenize import sent_tokenize
from utils import readDocx, readPdf, writeChunksToFile
nltk.download('punkt')
nltk.download('punkt_tab')

def splitTextIntoChunksV2(text, chunk_size=32):
    overlap= chunk_size // 8
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    print("Splitting Text into Chunks")
    
    # Tokenize the entire text
    tokens = tokenizer.tokenize(text)
    
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)

    return chunks

def splitTextIntoChunks(text, chunk_size=32):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    print("Splitting Text into Chunks")
    sentences = sent_tokenize(text) 
    
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        current_length += len(tokens)
        current_chunk.append(sentence)

        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunkSyllabus(file_name, output_path, chunk_size=32):
    _, file_extension = os.path.splitext(file_name)
    txt = ''
    
    if file_extension.lower() == '.docx':
        txt = readDocx(file_name)
    elif file_extension.lower() == '.pdf':
        txt = readPdf(file_name)
    else:
        raise ValueError("Unsupported file format. Only .docx and .pdf files are supported.")
    chunks = splitTextIntoChunks(txt, chunk_size)
    writeChunksToFile(chunks, output_path)
    return chunks

# def chunk_syllabi(file_list, chunk_size=32):
#     syllabus_name = 'syl_1'
#     for file in file_list:
#         print(file)
#         txt = read_docx("/Users/willsaccount/Desktop/Tailored Tutor/src/Syllabuses/" + file)
#         print("TXT FILE")
#         print(txt)
#         chunks = splitTextIntoChunks(txt, chunk_size)
#         write_chunks_to_file(chunks,"/Users/willsaccount/Desktop/Tailored Tutor/src/Chunked_Syllabuses/" + str(syllabus_name) + "_chunk_" + str(chunk_size) + ".json")
#         return chunks


# def main():
#     file_list = ["ACC 300 Syllabus FS24 2024 08 23.docx"]
#     file_path = "/Users/willsaccount/Desktop/Tailored Tutor/src/Syllabuses/ACC 300 Syllabus FS24 2024 08 23.docx"
#     if os.path.exists(file_path):
#         print(f"File {file_path} exists.")
#     else:
#         print(f"File {file_path} does not exist. Please check the path.")
    
#     txt = read_docx(file_path)
#     print(txt)
#     chunk_syllabus(file_list)



# if __name__ == "__main__":
#     main()