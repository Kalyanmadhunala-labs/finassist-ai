import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

register_vector(conn)

model = SentenceTransformer("all-MiniLM-L6-v2")

question = "What is Machine Learning?"

question_embedding = model.encode(question).astype("float32")

with conn.cursor() as cur:
    cur.execute(
        """
        SELECT chunk_id, source, content,
               1 - (embedding <=> %s) AS similarity
        FROM document_chunks
        WHERE source = 'ai_knowledge.txt'
        ORDER BY embedding <=> %s
        LIMIT 2
        """,
        (question_embedding, question_embedding)
    )

    results = cur.fetchall()

print("\nTop matching chunks:")

for row in results:
    chunk_id, source, content, similarity = row

    print(f"\nChunk ID: {chunk_id}")
    print(f"Source: {source}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Content: {content}")

conn.close()