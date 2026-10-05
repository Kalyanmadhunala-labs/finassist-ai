import re
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os


# Load environment variables
load_dotenv()


# Connect to PostgreSQL
conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

register_vector(conn)


# Read the knowledge document
with open("ai_knowledge.txt", "r", encoding="utf-8") as file:
    text = file.read()


# Split document into sentences
sentences = re.split(r'(?<=[.!?])\s+', text)


# Chunking settings
chunk_size = 200
chunk_overlap = 1

chunks = []
current_chunk = []


# Create chunks
for sentence in sentences:

    test_chunk = current_chunk + [sentence]
    test_text = " ".join(test_chunk)

    if len(test_text) <= chunk_size:
        current_chunk = test_chunk

    else:
        chunks.append(" ".join(current_chunk))

        overlap_sentences = current_chunk[-chunk_overlap:]

        current_chunk = overlap_sentences + [sentence]


if current_chunk:
    chunks.append(" ".join(current_chunk))


print("Number of chunks:", len(chunks))


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create embeddings
embeddings = model.encode(chunks).astype("float32")


print("Embedding shape:", embeddings.shape)


# Insert chunks + metadata + embeddings into PostgreSQL
with conn.cursor() as cur:

    for i, chunk in enumerate(chunks):

        cur.execute(
            """
            INSERT INTO document_chunks
            (chunk_id, source, content, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            (
                i,
                "ai_knowledge.txt",
                chunk,
                embeddings[i]
            )
        )


conn.commit()

print("All documents inserted successfully!")


conn.close()