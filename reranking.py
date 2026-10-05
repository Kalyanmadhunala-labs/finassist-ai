import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer, CrossEncoder
from dotenv import load_dotenv
import os

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

print("Connected to PostgreSQL")

# Models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

question = "What is Generative AI?"

print("\nQuestion:")
print(question)

# Convert question to embedding
question_embedding = embedding_model.encode(question).astype("float32")

# Retrieve candidate chunks
with conn.cursor() as cur:

    cur.execute(
        """
        SELECT chunk_id, content
        FROM document_chunks
        ORDER BY embedding <=> %s
        LIMIT 5
        """,
        (
            question_embedding,
        )
    )

    results = cur.fetchall()

# Prepare question-chunk pairs
pairs = []

for chunk_id, content in results:
    pairs.append([question, content])

# Rerank candidates
reranker_scores = reranker.predict(pairs)

# Combine chunk information with scores
ranked_results = []

for i, (chunk_id, content) in enumerate(results):

    ranked_results.append(
        (
            chunk_id,
            content,
            reranker_scores[i]
        )
    )

# Sort by reranker score
ranked_results.sort(
    key=lambda x: x[2],
    reverse=True
)

print("\nReranked Results:")

for chunk_id, content, score in ranked_results:

    print("\nChunk ID:", chunk_id)
    print("Reranker score:", round(float(score), 4))
    print("Content:", content)

conn.close()