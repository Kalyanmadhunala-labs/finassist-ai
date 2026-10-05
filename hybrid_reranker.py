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

# --------------------------------------------------
# 1. VECTOR SEARCH
# --------------------------------------------------

question_embedding = embedding_model.encode(question).astype("float32")

with conn.cursor() as cur:

    cur.execute(
        """
        SELECT chunk_id, content
        FROM document_chunks
        ORDER BY embedding <=> %s
        LIMIT 5
        """,
        (question_embedding,)
    )

    vector_results = cur.fetchall()


# --------------------------------------------------
# 2. KEYWORD SEARCH
# --------------------------------------------------

with conn.cursor() as cur:

    cur.execute(
        """
        SELECT chunk_id, content
        FROM document_chunks
        WHERE search_vector @@
              plainto_tsquery('english', %s)
        LIMIT 5
        """,
        (question,)
    )

    keyword_results = cur.fetchall()


# --------------------------------------------------
# 3. COMBINE CANDIDATES
# --------------------------------------------------

candidates = {}

for chunk_id, content in vector_results:
    candidates[chunk_id] = content

for chunk_id, content in keyword_results:
    candidates[chunk_id] = content


print("\nCandidate chunks:")

for chunk_id in candidates:
    print("Chunk ID:", chunk_id)


# --------------------------------------------------
# 4. RERANK CANDIDATES
# --------------------------------------------------

pairs = [
    [question, content]
    for content in candidates.values()
]

reranker_scores = reranker.predict(pairs)

ranked_results = []

for i, (chunk_id, content) in enumerate(candidates.items()):

    ranked_results.append(
        (
            chunk_id,
            content,
            float(reranker_scores[i])
        )
    )


# Highest score first
ranked_results.sort(
    key=lambda x: x[2],
    reverse=True
)


# --------------------------------------------------
# 5. DISPLAY FINAL RESULTS
# --------------------------------------------------

print("\nFinal Reranked Results:")

for chunk_id, content, score in ranked_results:

    print("\nChunk ID:", chunk_id)
    print("Reranker score:", round(score, 4))
    print("Content:", content)


conn.close()