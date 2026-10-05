import re
from sentence_transformers import SentenceTransformer

# Step 1: Read the document
with open("ai_knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Step 2: Split document into sentences
sentences = re.split(r'(?<=[.!?])\s+', text)

# Step 3: Chunk settings
chunk_size = 200
chunk_overlap = 1

# Step 4: Create chunks
chunks = []
current_chunk = []

for sentence in sentences:

    test_chunk = current_chunk + [sentence]
    test_text = " ".join(test_chunk)

    if len(test_text) <= chunk_size:
        current_chunk = test_chunk

    else:
        chunks.append(" ".join(current_chunk))

        overlap_sentences = current_chunk[-chunk_overlap:]

        current_chunk = overlap_sentences + [sentence]

# Step 5: Save the final chunk
if current_chunk:
    chunks.append(" ".join(current_chunk))

# Step 6: Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 7: Convert all chunks into embeddings
embeddings = model.encode(chunks)

# Step 8: Display basic information
print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("\nFirst chunk:")
print(chunks[0])

print("\nFirst chunk embedding:")
print(embeddings[0])

print("\nFirst chunk embedding length:")
print(len(embeddings[0]))
# Step 8: Display basic information
print("Number of chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)

print("\nFirst chunk:")
print(chunks[0])

print("\nFirst chunk embedding:")
print(embeddings[0])

print("\nFirst chunk embedding length:")
print(len(embeddings[0]))
from sklearn.metrics.pairwise import cosine_similarity

# Create an embedding for the user's question
question = "What is Machine Learning?"
question_embedding = model.encode([question])

# Compare the question with all document chunks
similarity_scores = cosine_similarity(
    question_embedding,
    embeddings
)

print("\nSimilarity scores:")
print(similarity_scores)
import numpy as np

scores = similarity_scores[0]

sorted_indices = np.argsort(scores)

print("Sorted indices:", sorted_indices)
sorted_indices = sorted_indices[::-1]

print("Sorted indices:", sorted_indices)
top_k = 2

top_indices = sorted_indices[:top_k]

print("Top indices:", top_indices)
print("\nRetrieved chunks:")

for index in top_indices:
    print("\nChunk", index + 1)
    print(chunks[index])
    context = "\n\n".join(chunks[index] for index in top_indices)

print("\nContext:")
print(context)
prompt = f"""
Answer the question using only the context provided below.

Context:
{context}

Question:
{question}

Answer:
"""

print("\nPrompt:")
print(prompt)