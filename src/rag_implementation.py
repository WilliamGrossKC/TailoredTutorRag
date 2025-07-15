from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import torch
import faiss
import numpy as np
import time
import nltk
import os
import docx
import json
from nltk.tokenize import sent_tokenize
from utils import readDocx, readJsonFile
nltk.download('punkt')
nltk.download('punkt_tab')

# Load BERT for basic RAG implementation
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

# Load Sentence Transformer for semantic search
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Re-ranking model for cross-encoder re-ranking
reranker_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")

TESTING = True

# Tokenize the text
def tokenizeText(text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, token_ids

# Embed the tokens using BERT (Basic RAG)
def embedTokens(token_ids):
    inputs = torch.tensor([token_ids])
    with torch.no_grad():
        embeddings = model(inputs)[0]  # (batch_size, sequence_length, hidden_size)
    embeddings_mean = embeddings.mean(dim=1)
    return embeddings_mean.squeeze(0)

# Store the embeddings in FAISS for basic RAG
def storeEmbeddingsInFaiss(embeddings_list):
    dim = embeddings_list[0].size(0)
    index = faiss.IndexFlatL2(dim) 
    embeddings_np = np.vstack([emb.numpy() for emb in embeddings_list]) 
    index.add(embeddings_np)
    return index

# Query the FAISS index (Basic RAG)
def queryIndex(index, query_embedding, k=8):
    query_embedding_np = query_embedding.numpy().reshape(1, -1)
    distances, indices = index.search(query_embedding_np, k)
    return indices, distances

# Semantic search using Sentence Transformers
def semanticSearch(chunks, query, k=8):
    chunk_embeddings = semantic_model.encode(chunks, convert_to_tensor=True)
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=k)
    indices = [[hit['corpus_id'] for hit in hit_list] for hit_list in hits]
    return indices, None

# TF-IDF search with cosine similarity
def tfidfSearch(chunks, query, k=8):
    vectorizer = TfidfVectorizer()
    chunk_tfidf_matrix = vectorizer.fit_transform(chunks)
    query_tfidf_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf_vector, chunk_tfidf_matrix)
    indices = similarities.argsort()[0][-k:][::-1]
    return indices, similarities[0, indices]

# BM25 search
def bm25Search(chunks, query, k=8):
    tokenized_chunks = [doc.split() for doc in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    indices = np.argsort(scores)[-k:][::-1]
    return indices, scores[indices]

# Re-ranking using Cross-Encoder
def rerankSearch(chunks, query, top_k_chunks, k=8):
    inputs = reranker_tokenizer([query + " [SEP] " + chunks[i] for i in top_k_chunks], return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = reranker_model(**inputs).logits
    ranked_indices = outputs.argsort(dim=0, descending=True).squeeze()[:k]
    return top_k_chunks[ranked_indices], outputs.squeeze()[ranked_indices]

# Full pipeline with multiple algorithms
def pipeline(chunks, query_text, chunk_size, k_value, algo_type):
    if algo_type == 0:
        # Basic RAG
        embeddings_list = []
        for chunk in chunks:
            tokens, token_ids = tokenizeText(chunk)
            embeddings = embedTokens(token_ids)
            embeddings_list.append(embeddings)
        index = storeEmbeddingsInFaiss(embeddings_list)
        _, query_token_ids = tokenizeText(query_text)
        query_embedding = embedTokens(query_token_ids)
        indices, distances = queryIndex(index, query_embedding, k_value)
        
    elif algo_type == 1:
        # Semantic Search
        indices, distances = semanticSearch(chunks, query_text, k=k_value)
        
    elif algo_type == 2:
        # TF-IDF Search
        indices, distances = tfidfSearch(chunks, query_text, k=k_value)
        
    elif algo_type == 3:
        # BM25 Search
        indices, distances = bm25Search(chunks, query_text, k=k_value)
        
    elif algo_type == 4:
        # Re-Ranking Search (use BM25 for initial coarse filtering)
        indices, _ = bm25Search(chunks, query_text, k=10)  # Get top 10 chunks
        indices, distances = rerankSearch(chunks, query_text, indices, k=k_value)
    
    return indices, distances

def flatten_indices(indices):
    # A helper function to flatten nested lists
    if isinstance(indices, (list, tuple)):  # Handle lists and tuples
        flat_list = []
        for item in indices:
            if isinstance(item, (list, tuple)):  # If it's a nested list or tuple, flatten it recursively
                flat_list.extend(flatten_indices(item))
            else:
                flat_list.append(item)
        return flat_list
    elif hasattr(indices, 'shape'):  # Handle NumPy arrays (or similar objects)
        return indices.flatten().tolist()  # Convert NumPy arrays to a flattened list
    else:
        return [indices]  # If it's a single value, return it as a list

# Main RAG function to test different algorithms
def ragAlgo(chunked_txt, auto_prompts, chunk_size=32, k_value=3, algo_type=0): 
    count = 0
    correct = 0
   
    for prompt in auto_prompts:
        indices, distances = pipeline(chunked_txt, prompt, chunk_size, k_value, algo_type)
        #result.append(("Rag context Chunk: " + str(indices) + " Versus Prompt Chunk: " + str(count) + "\n"))

        if TESTING:
            # print(prompt)
            # print("Distances: " + str(distances))
            # print(("Rag context Chunk: " + str(indices) + " Versus Prompt Chunk: " + str(count) + "\n"))
            matched = False
            flat_indices = flatten_indices(indices)
    
            # Now compare count with flattened indices
            try:
                if count in flat_indices:
                    correct += 1
                    matched = True
            except ValueError as e:
                print(f"Comparison error: {e}. indices={indices}, count={count}")
            if not matched:
                print(f"Weird edge case: indices={indices}, count={count}")

            # result.append(("Rag context Chunk: " + str(indices) + " Versus Prompt Chunk: " + str(count) + "\n"))
            count += 1
            
    if TESTING:
        with open("/Users/willsaccount/Desktop/Tailored Tutor/src/Results/algorithm_results.txt", "a") as file:  # Open the file in append mode
            file.write("RESULTS FOR ALGORITHM: " + str(chunk_size) + " " + str(k_value) + " " + str(algo_type) + "\n")
            # file.write(str(result) + "\n")
            file.write("Score: " + str(algo_type) + " " + str(int(correct)/int(count) * 100) + "%\n")
            file.write("\n")  # Add an extra newline for better readability

        print("RESULTS FOR ALGORITHM: " + str(chunk_size) + " " + str(k_value) + " " + str(algo_type))
        # print(result)
        print("Score: " + str(algo_type) + " " + str(int(correct)/int(count) * 100) + "%")
    return (int(correct)/int(count) * 100)
        



# def main():
#     txt = readDocx("Syllabuses/ACC 300 Syllabus FS24 2024 08 23.docx")
#     prompts = read_json_file("/Users/willsaccount/Desktop/Tailored Tutor/src/Auto_Gen_Test_Questions/syl_1_autoprompts_32.json")
#     chunked_txt = split_text_into_chunks(txt, chunk_size=32)
#     rag_pipeline(chunked_txt, prompts)

# if __name__ == "__main__":
#     main()


