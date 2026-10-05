import os

import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg import register_vector
from sentence_transformers import CrossEncoder, SentenceTransformer


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# 2. CONNECT TO POSTGRESQL
# ============================================================

conn = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

register_vector(conn)

print("Connected to PostgreSQL")


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded")


# ============================================================
# 4. LOAD RERANKER MODEL
# ============================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker model loaded")


# ============================================================
# 5. USER QUESTION
# ============================================================

question = "What is Generative AI?"

print("\nQuestion:")
print(question)


# ============================================================
# 6. CREATE QUESTION EMBEDDING
# ============================================================

question_embedding = embedding_model.encode(
    question
).astype("float32")


# ============================================================
# 7. VECTOR SEARCH
# ============================================================

with conn.cursor() as cur:

    cur.execute(
        """
        SELECT
            chunk_id,
            content
        FROM document_chunks
        ORDER BY embedding <=> %s
        LIMIT 5
        """,
        (question_embedding,)
    )

    vector_results = cur.fetchall()


# ============================================================
# 8. KEYWORD SEARCH
# ============================================================

with conn.cursor() as cur:

    cur.execute(
        """
        SELECT
            chunk_id,
            content
        FROM document_chunks
        WHERE search_vector @@
              plainto_tsquery('english', %s)
        LIMIT 5
        """,
        (question,)
    )

    keyword_results = cur.fetchall()


# ============================================================
# 9. COMBINE VECTOR + KEYWORD RESULTS
# ============================================================

candidates = {}

for chunk_id, content in vector_results:

    candidates[chunk_id] = content


for chunk_id, content in keyword_results:

    candidates[chunk_id] = content


print("\nCandidate chunks:")

for chunk_id in candidates:

    print("Chunk ID:", chunk_id)


# ============================================================
# 10. CREATE QUESTION-CHUNK PAIRS
# ============================================================

pairs = []

for content in candidates.values():

    pairs.append(
        [question, content]
    )


# ============================================================
# 11. RERANK CANDIDATES
# ============================================================

reranker_scores = reranker.predict(
    pairs
)


# ============================================================
# 12. ATTACH RERANKER SCORES
# ============================================================

ranked_results = []

candidate_items = list(
    candidates.items()
)

for i, (chunk_id, content) in enumerate(
    candidate_items
):

    score = float(
        reranker_scores[i]
    )

    ranked_results.append(
        (
            chunk_id,
            content,
            score
        )
    )


# ============================================================
# 13. SORT BY RERANKER SCORE
# ============================================================

ranked_results.sort(
    key=lambda x: x[2],
    reverse=True
)


# ============================================================
# 14. SELECT TOP RESULTS
# ============================================================

top_k = 2

top_results = ranked_results[:top_k]


# ============================================================
# 15. DISPLAY TOP RETRIEVED CHUNKS
# ============================================================

print("\nTop reranked chunks:")

for chunk_id, content, score in top_results:

    print("\nChunk ID:", chunk_id)

    print(
        "Reranker score:",
        round(score, 4)
    )

    print(
        "Content:",
        content
    )


# ============================================================
# 16. BUILD CONTEXT
# ============================================================

context_parts = []

for chunk_id, content, score in top_results:

    context_parts.append(
        content
    )


context = "\n\n".join(
    context_parts
)


# ============================================================
# 17. BUILD LLM PROMPT
# ============================================================

prompt = f"""
Answer the question using only the context provided below.

If the answer is not present in the context,
say that the information is not available.

Context:
{context}

Question:
{question}

Answer:
"""


# ============================================================
# 18. CONNECT TO GROQ
# ============================================================

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


# ============================================================
# 19. SEND CONTEXT + QUESTION TO GROQ
# ============================================================

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)


# ============================================================
# 20. GET FINAL ANSWER
# ============================================================

final_answer = (
    response
    .choices[0]
    .message
    .content
)


# ============================================================
# 21. DISPLAY FINAL ANSWER
# ============================================================

print("\nFinal Answer:")
print(final_answer)


# ============================================================
# 22. CLOSE DATABASE CONNECTION
# ============================================================

conn.close()

print("\nDatabase connection closed.")